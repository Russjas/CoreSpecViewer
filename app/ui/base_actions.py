"""
Base class for action handlers.

Provides common infrastructure for ribbon registration and context management.
"""

from PyQt5.QtWidgets import QMessageBox

from ..models import CurrentContext


class BaseActions:
    """
    Base class for all action handlers.
    
    Action handlers encapsulate related operations and their ribbon UI.
    Each handler:
    - Holds a reference to the shared CurrentContext
    - Registers its buttons/menus with the ribbon
    - Implements callback methods for user actions
    
    Subclasses must implement stage_ribbon() to define their UI.
    """
    
    def __init__(self, context: CurrentContext, ribbon, parent=None):
        """
        Initialize the action handler.
        
        Args:
            context: Shared application context (will be refreshed by main controller)
            ribbon: Ribbon interface for registering buttons (expects FlexibleRibbon API)
            parent: Parent widget for dialogs (typically MainRibbonController)
        """
        self.cxt = context
        self.ribbon = ribbon
        self.controller = parent
        self.stage_ribbon()
    
    def stage_ribbon(self):
        """
        Define and register ribbon structure.
        
        Subclasses must override this method and call self._register_tab() 
        and/or self._register_group() to define their UI.
        
        Example:
            def stage_ribbon(self):
                # Add to default tab
                self._register_group('My Group', [
                    ("button", "Action 1", self.action_1, "Tooltip for action 1"),
                    ("button", "Action 2", self.action_2, "Tooltip for action 2"),
                ])
                
                # Or create a new tab
                self._register_tab('Advanced')
                self._register_group('Advanced Tools', [
                    ("button", "Advanced Action", self.advanced, "Tooltip"),
                ])
        """
        raise NotImplementedError("Subclasses must implement stage_ribbon()")
    
    def _register_tab(self, tab_name: str):
        """
        Create a new ribbon tab.
        
        All subsequent _register_group() calls will add groups to this tab
        until another tab is created.
        
        This is the ONLY method that knows about the tab creation API.
        
        Args:
            tab_name: Name of the tab to create
        """
        self.ribbon.add_tab(tab_name)
    
    def _register_group(self, group_name: str, entries: list):
        """
        Register a group of buttons with the ribbon.
        
        This is the ONLY method that knows about the group registration API.
        If the ribbon implementation changes, only modify these two methods.
        
        Adds the group to the current active tab. To add to a different tab,
        call self._register_tab(tab_name) first.
        
        Args:
            group_name: Label for the button group
            entries: List of button/menu specifications in the format:
                     ("button", label, callback, tooltip)
                     ("menu", label, submenu_list)
                     ("menu", label, submenu_list, tooltip)
        """
        self.ribbon.add_group(group_name, entries)
    
    # ============ Helper Utilities ============
    
    def _show_error(self, title: str, message: str):
        """Show a consistent error dialog."""
        QMessageBox.warning(self.controller, title, message)
    
    def _show_info(self, title: str, message: str):
        """Show a consistent info dialog."""
        QMessageBox.information(self.controller, title, message)