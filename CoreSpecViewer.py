# -*- coding: utf-8 -*-
"""
Standalone launcher for the CoreSpecViewer application.

This script exists as a convenience entry point so that end-users can start the
GUI simply by running:

    python CoreSpecViewer.py

It performs no application logic itself. Instead, it imports the public
`main()` function from the `app` package and delegates the full startup
sequence to it.

The actual application structure (Qt windows, pages, models, configuration,
and tool logic) lives entirely under the `app/` package. This file is just a
thin wrapper to avoid requiring users to know or invoke:

    python -m app.main

Developers may safely ignore this file and work directly with the `app`
package.
"""

# CoreSpecViewer.py
from app.main import main

if __name__ == "__main__":
    main()