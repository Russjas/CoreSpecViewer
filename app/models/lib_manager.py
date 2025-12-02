"""
Non-UI manager for spectral library databases and collections.

This class centralises:
    - Opening/closing a SQLite spectral library (via QSqlDatabase)
    - Exposing a QSqlTableModel for the Samples table (for Qt views)
    - Managing in-memory collections of SampleIDs
    - Building exemplar spectra for collections
    - Exporting a subset of the DB based on a collection
    - Utility queries (fetch one spectrum, filter by band coverage)

UI widgets (LibraryPage, ClusterWindow, etc.) should:
    - Hold a reference to a shared LibraryManager
    - Use its model for their QTableView
    - Call its methods to manipulate collections / exemplars
"""

from __future__ import annotations

import os
import sqlite3
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np
from PyQt5.QtCore import QByteArray
from PyQt5.QtSql import QSqlDatabase, QSqlQuery, QSqlTableModel

# Mirror the constants you currently have in lib_page.py
SAMPLE_TABLE_NAME = "Samples"
SPECTRA_TABLE_NAME = "Spectra"
WAVELENGTH_BLOB_COL = "XData"
REFLECTANCE_BLOB_COL = "YData"

# NumPy dtype for 4-byte (32-bit) little-endian float BLOBs
BLOB_DTYPE = "<f4"

# Column positions in the Samples table model
ID_COLUMN_INDEX = 0    # SampleID
NAME_COLUMN_INDEX = 1  # Name


class LibraryManager:
    """
    Headless manager for spectral libraries and exemplar collections.

    Responsibilities
    ----------------
    - Owns the QSqlDatabase connection and QSqlTableModel for Samples
    - Owns in-memory collections: name -> set[SampleID]
    - Provides exemplar building: collection -> {sid: (label, x_nm, y)}
    - Provides band-coverage filtering: given a bands array (nm), return SampleIDs
    - Provides single-spectrum fetch: SampleID -> (x_nm, y)

    No UI: no QFileDialog, QMessageBox, etc.
    Callers are expected to:
      - Decide which DB path to open
      - Handle errors / messages
      - Ask this manager for raw data to display / plot.
    """

    def __init__(self) -> None:
        # DB + model
        self.db_path: Optional[str] = None
        self.db: QSqlDatabase | None = None
        self.model: QSqlTableModel | None = None

        # Collections: collection_name -> set of SampleIDs
        self.collection_ids = set()
        self.collections: dict[str, set[int]] = {}
        self.exemplars_by_collection = {}
        self.exemplars = {}

    # ------------------------------------------------------------------
    # DB lifecycle
    # ------------------------------------------------------------------
    

    def open_database(self, path: str) -> QSqlTableModel:
        """
        (Re)open a SQLite DB file and bind the QSqlTableModel to it.

        Parameters
        ----------
        path : str
            Path to the SQLite DB file.

        Returns
        -------
        QSqlTableModel
            Model bound to the Samples table, suitable for attaching to a view.

        Raises
        ------
        RuntimeError
            If opening the DB or binding the model fails.
        """
        # If already open on the same path, just return existing model
        if self.db_path and os.path.abspath(self.db_path) == os.path.abspath(path):
            if self.model is not None:
                return self.model

        self.close_database()

        conn_name = f"libmgr_{id(self)}"
        db = QSqlDatabase.addDatabase("QSQLITE", conn_name)
        db.setDatabaseName(path)

        if not db.open():
            QSqlDatabase.removeDatabase(conn_name)
            raise RuntimeError(f"Failed to open SQLite DB: {path}")

        self.db_path = path
        self.db = db

        # Basic schema check
        if not self._table_exists(SAMPLE_TABLE_NAME) or not self._table_exists(SPECTRA_TABLE_NAME):
            db.close()
            QSqlDatabase.removeDatabase(conn_name)
            self.db = None
            self.db_path = None
            raise RuntimeError("DB does not have required tables 'Samples' and 'Spectra'.")

        # Bind model to Samples table
        model = QSqlTableModel(None, db)
        model.setTable(SAMPLE_TABLE_NAME)
        if not model.select():
            name = model.lastError().text()
            db.close()
            QSqlDatabase.removeDatabase(conn_name)
            self.db = None
            self.db_path = None
            raise RuntimeError(f"Failed to select Samples table: {name}")
        
        self.model = model
        self.collection_ids.clear()
        self.collections.clear()
        self.exemplars_by_collection.clear()
        
        return model

    def close_database(self) -> None:
        """Close any open DB connection and drop the model."""
        # Drop model reference
        if self.model is not None:
            self.model.deleteLater()
            self.model = None

        # Close DB connection
        if self.db is not None and self.db.isValid():
            conn_name = self.db.connectionName()
            if self.db.isOpen():
                self.db.close()
            QSqlDatabase.removeDatabase(conn_name)

        self.db = None
        self.db_path = None
        self.collections.clear()
        self.exemplars_by_collection.clear()
        self.exemplars.clear()

    def get_model(self) -> QSqlTableModel | None:
        """Return the QSqlTableModel bound to Samples, for a QTableView."""
        return self.model

    def _table_exists(self, name: str) -> bool:
        if not self.db:
            return False
        q = QSqlQuery(self.db)
        ok = q.exec_(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{name}'")
        if not ok:
            return False
        return q.next()

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------
    def list_collections(self) -> list[str]:
        """Return sorted list of known collection names."""
        return sorted(self.collections.keys())

    def get_collection_ids(self, name: str) -> Set[int]:
        """Return the SampleID set for a collection (empty set if unknown)."""
        return self.collections.get(name, set())

    def set_collection_ids(self, name: str, ids: Iterable[int]) -> None:
        """Replace collection contents with the given SampleIDs."""
        self.collections[name] = set(int(v) for v in ids)

    def add_to_collection(self, name: str, ids: Iterable[int]) -> Tuple[int, int]:
        """
        Add SampleIDs to a collection.

        Returns
        -------
        added : int
            Number of newly added IDs.
        total : int
            Total IDs now in the collection.
        """
        coll = self.collections.setdefault(name, set())
        before = len(coll)
        coll.update(int(v) for v in ids)
        added = len(coll) - before
        return added, len(coll)

    def clear_collection(self, name: str) -> None:
        """Remove a collection by name (no-op if not present)."""
        self.collections.pop(name, None)

    # ------------------------------------------------------------------
    # Exemplars
    # ------------------------------------------------------------------
    def get_collection_exemplars(
        self,
        name: str
    ) -> Dict[int, Tuple[str, np.ndarray, np.ndarray]]:
        """
        Build or fetch cached exemplar spectra for a named collection.

        Parameters
        ----------
        name : str
            Collection name.

        Returns
        -------
        dict[int, (label, x_nm, y)]
            Mapping SampleID -> (human-readable label, wavelengths in nm, reflectance).
        """
        if not self.is_open():
            raise RuntimeError("No library DB is open.")

        # Cached?
        if name in self.exemplars_by_collection:
            return self.exemplars_by_collection[name]

        ids = list(self.collections.get(name, set()))
        if not ids:
            # Return empty; UI decides how to message "empty collection"
            self.exemplars_by_collection[name] = {}
            return {}

        db = self.db
        if db is None:
            return {}

        # Map SampleID -> Name
        id_to_name: Dict[int, str] = {}
        q = QSqlQuery(db)
        placeholders = ",".join("?" for _ in ids)
        q.prepare(f"SELECT SampleID, Name FROM {SAMPLE_TABLE_NAME} WHERE SampleID IN ({placeholders})")
        for v in ids:
            q.addBindValue(int(v))

        if q.exec_():
            while q.next():
                sid = int(q.value(0))
                name_str = str(q.value(1))
                id_to_name[sid] = name_str

        exemplars: Dict[int, Tuple[str, np.ndarray, np.ndarray]] = {}
        q2 = QSqlQuery(db)
        q2.prepare(
            f"SELECT SampleID, {WAVELENGTH_BLOB_COL}, {REFLECTANCE_BLOB_COL} "
            f"FROM {SPECTRA_TABLE_NAME} WHERE SampleID IN ({placeholders})"
        )
        for v in ids:
            q2.addBindValue(int(v))

        if not q2.exec_():
            # Log/inspect q2.lastError() in calling code if needed
            self.exemplars_by_collection[name] = {}
            return {}

        while q2.next():
            sid = int(q2.value(0))
            x_bytes = q2.value(1)
            y_bytes = q2.value(2)
            if x_bytes is None or y_bytes is None:
                continue

            x = np.frombuffer(bytes(x_bytes), dtype=BLOB_DTYPE)
            y = np.frombuffer(bytes(y_bytes), dtype=BLOB_DTYPE)
            # Library X is stored in µm → convert to nm
            x_nm = (x * 1000.0).astype(np.float32)
            label = id_to_name.get(sid, f"ID{sid}")
            exemplars[sid] = (label, x_nm, y.astype(np.float32))

        self.exemplars_by_collection[name] = exemplars
        return exemplars

    # ------------------------------------------------------------------
    # Export subset DB
    # ------------------------------------------------------------------
    def export_collection_to_db(
        self,
        collection_name: str,
        dst_path: str,
        samples_table: str = SAMPLE_TABLE_NAME,
        spectra_table: str = SPECTRA_TABLE_NAME,
        spectra_fk_col: str = "SampleID",
    ) -> None:
        """
        Export a collection to a new SQLite DB at dst_path.

        This clones schema and copies:
          - rows from `samples_table` where SampleID in collection
          - rows from `spectra_table` where spectra_fk_col in collection

        Raises
        ------
        RuntimeError on failure.
        """
        if not self.db_path:
            raise RuntimeError("No source DB path is known.")
        ids_set = self.collections.get(collection_name, set())
        if not ids_set:
            raise RuntimeError(f"Collection '{collection_name}' is empty.")

        src_path = self.db_path
        samples_pk_col = self._column_name_by_index(samples_table, ID_COLUMN_INDEX)

        try:
            self._export_subset_db(
                src_path=src_path,
                dst_path=dst_path,
                sample_ids=list(ids_set),
                samples_table=samples_table,
                spectra_table=spectra_table,
                spectra_fk_col=spectra_fk_col,
                samples_pk_col=samples_pk_col,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to export subset DB: {e}") from e

    def _column_name_by_index(self, table_name: str, col_index: int) -> str:
        if not self.db:
            raise RuntimeError("No DB open.")
        q = QSqlQuery(self.db)
        if not q.exec_(f"PRAGMA table_info({table_name});"):
            raise RuntimeError(q.lastError().text())
        cols = []
        while q.next():
            cols.append(q.value(1))  # name
        if col_index < 0 or col_index >= len(cols):
            raise IndexError(f"Column index {col_index} out of range for {table_name}")
        return cols[col_index]

    def _export_subset_db(
        self,
        src_path: str,
        dst_path: str,
        sample_ids: Iterable[int],
        samples_table: str,
        spectra_table: str,
        spectra_fk_col: str,
        samples_pk_col: str,
    ) -> None:
        """
        Create a new SQLite DB with identical schema, then insert subset rows:
        - All rows from `samples_table` where samples_pk_col IN sample_ids
        - All rows from `spectra_table` where spectra_fk_col IN sample_ids
        Also copies indices, triggers, and views.
        """
        sample_ids = [int(v) for v in sample_ids]

        # Safety: ensure path is new/empty
        if os.path.exists(dst_path):
            os.remove(dst_path)

        src = sqlite3.connect(src_path)
        dst = sqlite3.connect(dst_path)

        try:
            src.row_factory = sqlite3.Row
            s_cur = src.cursor()
            d_cur = dst.cursor()

            # 1) Clone schema (tables, indices, triggers, views)
            schema_rows = s_cur.execute(
                "SELECT type, name, sql FROM sqlite_master "
                "WHERE name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
                "ORDER BY CASE type "
                " WHEN 'table' THEN 0 WHEN 'index' THEN 1 WHEN 'trigger' THEN 2 "
                " WHEN 'view' THEN 3 ELSE 4 END, name;"
            ).fetchall()

            dst.execute("PRAGMA foreign_keys=OFF;")
            dst.execute("BEGIN;")
            for row in schema_rows:
                sql = row["sql"]
                if not sql:
                    continue
                d_cur.execute(sql)
            dst.commit()

            # 2) Insert subset rows
            dst.execute("BEGIN;")

            # Helpers to introspect columns
            def table_cols(conn, tname):
                cur = conn.execute(f"PRAGMA table_info({tname});")
                return [r[1] for r in cur.fetchall()]

            samples_cols = table_cols(src, samples_table)
            spectra_cols = table_cols(src, spectra_table)

            ids_placeholders = ",".join("?" for _ in sample_ids)

            # Samples subset
            sample_rows = s_cur.execute(
                f"SELECT * FROM {samples_table} "
                f"WHERE {samples_pk_col} IN ({ids_placeholders})",
                sample_ids,
            ).fetchall()

            spectra_rows = s_cur.execute(
                f"SELECT * FROM {spectra_table} "
                f"WHERE {spectra_fk_col} IN ({ids_placeholders})",
                sample_ids,
            ).fetchall()

            ins_samples = f"INSERT INTO {samples_table} ({', '.join(samples_cols)}) VALUES ({', '.join('?' for _ in samples_cols)})"
            ins_spectra = f"INSERT INTO {spectra_table} ({', '.join(spectra_cols)}) VALUES ({', '.join('?' for _ in spectra_cols)})"

            for row in sample_rows:
                d_cur.execute(ins_samples, [row[c] for c in samples_cols])

            for row in spectra_rows:
                d_cur.execute(ins_spectra, [row[c] for c in spectra_cols])

            dst.commit()
        except Exception:
            dst.rollback()
            raise
        finally:
            src.close()
            dst.close()

    # ------------------------------------------------------------------
    # Spectrum / band utilities
    # ------------------------------------------------------------------
    def get_spectrum(self, sample_id: int) -> Tuple[np.ndarray, np.ndarray]:
        print(sample_id)
        """
        Fetch a single spectrum from the Spectra table.

        Returns
        -------
        x_nm : np.ndarray
            Wavelengths in nm.
        y : np.ndarray
            Reflectance values (as stored, typically unitless).
        """
        if not self.is_open():
            raise RuntimeError("No library DB is open.")
        db = self.db
        if db:
            print('db')
        assert db is not None

        q = QSqlQuery(db)
        sql = (
            f"SELECT {WAVELENGTH_BLOB_COL}, {REFLECTANCE_BLOB_COL} "
            f"FROM {SPECTRA_TABLE_NAME} WHERE SampleID = {int(sample_id)};"
        )
        if not q.exec_(sql):
            raise RuntimeError(f"Failed to fetch spectrum for SampleID={sample_id}: {q.lastError().text()}")

        if not q.next():
            raise KeyError(f"No spectrum row found for SampleID={sample_id}")

        x_bytes = q.value(0)
        y_bytes = q.value(1)
        if x_bytes is None or y_bytes is None:
            raise RuntimeError(f"Null BLOB(s) for SampleID={sample_id}")

        x = np.frombuffer(bytes(x_bytes), dtype=BLOB_DTYPE)
        y = np.frombuffer(bytes(y_bytes), dtype=BLOB_DTYPE)
        x_nm = (x * 1000.0).astype(np.float32)
        return x_nm, y.astype(np.float32)

    def get_sample_name(self, sample_id: int) -> Optional[str]:
        """
        Fetch the 'Name' field for a sample, if present.
        """
        if not self.is_open():
            raise RuntimeError("No library DB is open.")
        db = self.db
        assert db is not None

        q = QSqlQuery(db)
        sql = f"SELECT Name FROM {SAMPLE_TABLE_NAME} WHERE SampleID = {int(sample_id)};"
        if not q.exec_(sql):
            return None
        if not q.next():
            return None
        val = q.value(0)
        return None if val is None else str(val)

    def filter_ids_covering_bands(self, bands_nm: np.ndarray) -> Set[int]:
        """
        Return SampleIDs whose XData fully covers the given band range.

        Parameters
        ----------
        bands_nm : np.ndarray
            1D array of wavelengths in nm (from current_obj.bands).

        Returns
        -------
        set[int]
            SampleIDs whose wavelength coverage fully includes [min(bands_nm), max(bands_nm)].
        """
        if not self.is_open():
            raise RuntimeError("No library DB is open.")
        db = self.db
        assert db is not None

        if bands_nm is None or bands_nm.size == 0:
            return set()

        target_min_nm = float(np.nanmin(bands_nm))
        target_max_nm = float(np.nanmax(bands_nm))
        target_min_um = target_min_nm / 1000.0
        target_max_um = target_max_nm / 1000.0

        ok_ids: Set[int] = set()

        q = QSqlQuery(db)
        if not q.exec_(f"SELECT SampleID, {WAVELENGTH_BLOB_COL} FROM {SPECTRA_TABLE_NAME}"):
            # On failure, return empty and let caller inspect q.lastError()
            return set()

        while q.next():
            sid = int(q.value(0))
            x_bytes = q.value(1)
            if x_bytes is None:
                continue
            try:
                x_um = np.frombuffer(bytes(x_bytes), dtype=BLOB_DTYPE)
            except Exception:
                continue
            if x_um.size == 0:
                continue

            if np.nanmin(x_um) <= target_min_um and np.nanmax(x_um) >= target_max_um:
                ok_ids.add(sid)

        return ok_ids

        
    def is_open(self) -> bool:
        """Return True if a DB connection is open and valid."""
        return self.db is not None and self.db.isValid() and self.db.isOpen()
    
    #=======Methods for adding to DB
    def add_sample(
        self,
        name: str,
        wavelengths_nm: np.ndarray,
        reflectance: np.ndarray,
        metadata: Optional[Dict[str, any]] = None
    ) -> int:
        """
        Add a new sample to the library database.
    
        Parameters
        ----------
        name : str
            Sample name (required).
        wavelengths_nm : np.ndarray
            Wavelength array in nanometers.
        reflectance : np.ndarray
            Reflectance values corresponding to wavelengths.
        metadata : dict, optional
            Additional column values for the Samples table.
            Keys should match column names in the Samples table.
    
        Returns
        -------
        int
            The SampleID of the newly inserted sample.
    
        Raises
        ------
        RuntimeError
            If no DB is open or insertion fails.
        ValueError
            If wavelengths and reflectance arrays have different lengths.
        """
        if not self.is_open():
            raise RuntimeError("No library DB is open.")
        
        if len(wavelengths_nm) != len(reflectance):
            raise ValueError(
                f"Wavelength and reflectance arrays must have same length: "
                f"{len(wavelengths_nm)} vs {len(reflectance)}"
            )
        
        db = self.db
        assert db is not None
    
        # Convert nm to µm for storage (matching your library format)
        wavelengths_um = np.array(wavelengths_nm / 1000.0).astype(np.float32)
        reflectance = reflectance.astype(np.float32)
    
        # Convert to BLOB format
        x_blob = np.array(wavelengths_um).tobytes()
        y_blob = np.array(reflectance).tobytes()
        
    
        try:
            # Insert into Samples table
            q = QSqlQuery(db)
            
            # Build column list and values
            cols = ["Name"]
            vals = [name]
            
            if metadata:
                for key, value in metadata.items():
                    if key.lower() != "sampleid":  # Skip ID, it's auto-generated
                        cols.append(key)
                        vals.append(value)
            
            placeholders = ", ".join("?" for _ in vals)
            col_str = ", ".join(cols)
            
            q.prepare(f"INSERT INTO {SAMPLE_TABLE_NAME} ({col_str}) VALUES ({placeholders})")
            for val in vals:
                q.addBindValue(val)
            
            if not q.exec_():
                raise RuntimeError(f"Failed to insert sample: {q.lastError().text()}")
            
            # Get the newly created SampleID
            sample_id = q.lastInsertId()
            if sample_id is None:
                raise RuntimeError("Failed to retrieve new SampleID")
            sample_id = int(sample_id)
            
            # Insert into Spectra table
            q2 = QSqlQuery(db)
            q2.prepare(
                f"INSERT INTO {SPECTRA_TABLE_NAME} "
                f"(SampleID, {WAVELENGTH_BLOB_COL}, {REFLECTANCE_BLOB_COL}) "
                f"VALUES (?, ?, ?)"
            )
            q2.addBindValue(sample_id)
            q2.addBindValue(QByteArray(x_blob))
            q2.addBindValue(QByteArray(y_blob))
            
            if not q2.exec_():
                # Rollback - delete the sample we just inserted
                q3 = QSqlQuery(db)
                q3.exec_(f"DELETE FROM {SAMPLE_TABLE_NAME} WHERE SampleID = {sample_id}")
                raise RuntimeError(f"Failed to insert spectrum: {q2.lastError().text()}")
            
            # Refresh the model to show the new entry
            if self.model:
                self.model.select()
            
            return sample_id
            
        except Exception as e:
            raise RuntimeError(f"Failed to add sample to library: {e}") from e
    
    
    