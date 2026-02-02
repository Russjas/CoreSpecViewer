
"""
band_maths
==========

Safe evaluation of spectral band-math expressions with function support.

This module lets users define band-math operations as strings with wavelength
references (R1400 or R(1400)) and safe functions like interp(). Bare numeric 
literals are treated as constants.

Core behaviour
--------------

Expressions support:
- Wavelength references: R2300 or R(2300)
- Arithmetic operators: +, -, *, /, **
- Safe functions: interp(R1300, R1500) for linear interpolation
- Numeric constants: 1, 0.5, 1200

Examples:
- "(R2300 - R2200) / (R2000 - R1400) + 1200"
- "1 - R(1400) / interp(R(1300), R(1500))"

Safety and sanitisation
-----------------------

1. Expressions are sanitized to allow only safe characters
2. AST parsing validates Python syntax
3. FunctionValidator checks only allowed functions are used
4. WavelengthExtractor finds all wavelength references
5. Evaluation in restricted namespace with only approved functions

Typical usage
-------------

.. code-block:: python

    from app.spectral_ops import band_maths

    expr = "1 - R(1400) / interp(R(1300), R(1500))"

    result = band_maths.evaluate_expression(
        expr=expr,
        data=cube,          # shape (H, W, B)
        wavelengths=bands   # shape (B,)
    )
"""


import re
import ast
import numpy as np
import logging

logger = logging.getLogger(__name__)


# Allowed function names
ALLOWED_FUNCTIONS = {'interp', 'R'}


class FunctionValidator(ast.NodeVisitor):
    """
    Validates that only allowed functions are called in the expression.
    Raises ValueError if unauthorized functions are found.
    """
    def __init__(self):
        self.invalid_functions = []
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in ALLOWED_FUNCTIONS:
                self.invalid_functions.append(func_name)
        self.generic_visit(node)



class WavelengthSubstitutor(ast.NodeTransformer):
    """
    Replaces R(wavelength) calls with band reference identifiers.
    """
    def __init__(self):
        self.wavelengths = {}  # Maps identifier -> wavelength value
        self._counter = 0
    
    def visit_Call(self, node):
        # Check if this is an R() call
        if isinstance(node.func, ast.Name) and node.func.id == 'R':
            # Should have exactly one argument that's a number
            if len(node.args) == 1 and isinstance(node.args[0], ast.Constant):
                wavelength = float(node.args[0].value)
                identifier = f'_band_{self._counter}'
                self.wavelengths[identifier] = wavelength
                self._counter += 1
                
                # Replace R(wavelength) with identifier
                return ast.Name(id=identifier, ctx=ast.Load())
        
        # Recursively visit child nodes
        self.generic_visit(node)
        return node


def sanitize_expression(expr):
    """
    Sanitize and validate the input expression.
    Returns cleaned expression or None if invalid.
    """
    if not expr or not expr.strip():
        return None
        
    # Remove whitespace
    expr = re.sub(r'\s+', '', expr.strip())
    
    # Coarse lexical whitelist — allow only arithmetic symbols,
    # band references (Rxxxx), and approved function calls.
    # Commas permitted for interp(a, b[, weight]) arguments.
    if not re.match(r'^[0-9+\-*/(),.Ra-z_]+$', expr):
        return None
        
    return expr


def preprocess_r_prefix(expr):
    """
    Convert R1400 style notation to R(1400) function call notation.
    This allows both R1400 and R(1400) syntax.
    """
    # Match R followed by numbers (not already in parentheses)
    # Negative lookahead (?!\() ensures we don't match R(
    pattern = r'R(?!\()(\d+\.?\d*)'
    result = re.sub(pattern, r'R(\1)', expr)
    return result


def interp_function(band1, band2, weight=0.5):
    """
    Linear interpolation between two bands.

    Parameters
    ----------
    band1 : array-like
        First band image (H, W).
    band2 : array-like
        Second band image (H, W).
    weight : float or array-like, optional
        Interpolation weight in [0, 1], where:
            0.0 -> returns band1
            1.0 -> returns band2
            0.5 -> midpoint between band1 and band2 (default).
        Can be a scalar or a broadcastable array.

    Returns
    -------
    out : ndarray
        Interpolated image with shape (H, W).
    """
    a = np.asarray(band1, dtype=float)
    b = np.asarray(band2, dtype=float)
    w = np.asarray(weight, dtype=float)

    # Basic shape sanity check (let NumPy broadcasting do the rest)
    if a.shape != b.shape:
        raise ValueError(f"interp: band shape mismatch {a.shape} vs {b.shape}")

    return a + (b - a) * w

def parse_and_transform_expression(expr):
    """
    Parse expression and transform wavelength references.
    
    Returns:
        Tuple of (transformed_tree, wavelength_map) or (None, None) if invalid.
    """
    expr = sanitize_expression(expr)
    
    if not expr:
        logger.debug("No Expression after sanitisation")
        return None, None
    
    # Convert R1400 to R(1400) format
    expr = preprocess_r_prefix(expr)
    
    try:
        # Parse the expression
        tree = ast.parse(expr, mode='eval')
        
        # Validate functions
        validator = FunctionValidator()
        validator.visit(tree)
        if validator.invalid_functions:
            raise ValueError(f"Unauthorized functions: {', '.join(validator.invalid_functions)}")
        
        # Transform R() calls to band identifiers
        substitutor = WavelengthSubstitutor()
        transformed_tree = substitutor.visit(tree)
        
        return transformed_tree, substitutor.wavelengths
        
    except SyntaxError as e:
        return None, None


def evaluate_expression(expr, data, wavelengths):
    """
    Evaluate expression using AST.
        
    Args:
        expr: String expression to evaluate
        data: Numpy array with shape (H, W, B) containing spectral data
        wavelengths: Numpy array with shape (B,) containing wavelength values
    
    Returns:
        Calculated result: numpy array with shape (H, W)
        
    Raises:
        ValueError: If expression is invalid or result shape doesn't match expected (H, W)
    """
    tree, wavelength_map = parse_and_transform_expression(expr)
    if tree is None:
        raise ValueError("Invalid expression syntax")
    
    # Create restricted namespace with only allowed functions
    namespace = {
        "__builtins__": {},
        "interp": interp_function,
    }
    
    if data is not None and wavelengths is not None and wavelength_map:
        # Map each wavelength identifier to its nearest band
        for identifier, wavelength in wavelength_map.items():
            band_idx = int(np.argmin(np.abs(wavelengths - wavelength)))
            namespace[identifier] = data[:, :, band_idx]
    
    # Compile and evaluate
    try:
        ast.fix_missing_locations(tree)
        code = compile(tree, '<string>', 'eval')
        result = eval(code, namespace, {})
        
        # Convert scalar results to array for consistent output
        if np.isscalar(result):
            result = np.full(data.shape[:2], result)
        
        # Validate result shape
        expected_shape = data.shape[:2]
        if hasattr(result, 'shape') and result.shape != expected_shape:
            raise ValueError(
                f"Result shape {result.shape} doesn't match expected shape {expected_shape}. "
                f"Did you forget an R prefix on a wavelength reference?"
            )
        
        return result
        
    except Exception as e:
        raise ValueError(f"Evaluation error: {e}")
        
        
if __name__ == "__main__":
    from app.models.processed_object import ProcessedObject
    PO = ProcessedObject.from_path('D:/#A1_newhole/00_13-17-045-21w4_00_4957_4819m00_4824m00_2022-01-25_14-41-47_bands.npy')
    expr = "R2200 / interp(R2100, R2300)"
    result = evaluate_expression(expr, PO.savgol, PO.bands)
        
