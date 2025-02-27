from pathlib import Path
import os
import ipywidgets as widgets
from IPython.display import display


class MusicFileManager:
    """Handles music file selection and management through a UI interface"""

    def __init__(self, base_path="./music_database"):
        # self.base_path = Path(base_path)
        self.base_path = base_path
        self.files = self._initialize_file_mapping()
        self.filepath_dropdown = None
        self.path_display = None
        self._setup_widgets()

    def _initialize_file_mapping(self):
        """Initialize the mapping of display names to file paths"""
        files_dict = {}

        # Process files in the root directory
        for filename in os.listdir(self.base_path):
            full_path = os.path.join(self.base_path, filename)

            # If it's a file, add it directly
            if os.path.isfile(full_path):
                files_dict[filename] = full_path

            # If it's a directory, process its contents
            elif os.path.isdir(full_path):
                subdir_name = filename
                subdir_path = full_path

                # Process files in the immediate subdirectory
                for subfile in os.listdir(subdir_path):
                    sub_full_path = os.path.join(subdir_path, subfile)

                    # If it's a file, add it with its path
                    if os.path.isfile(sub_full_path):
                        files_dict[subfile] = sub_full_path

                    # If it's a directory, just add the directory path (don't go deeper)
                    elif os.path.isdir(sub_full_path):
                        files_dict[subfile] = sub_full_path

        return files_dict

    def _setup_widgets(self):
        """Setup the UI widgets for file selection"""
        self.filepath_dropdown = widgets.Dropdown(
            options=[(name, str(path)) for name, path in self.files.items()],
            description='Select piece:',
            style={'description_width': 'initial'},
            layout={'width': '500px'}
        )

        self.path_display = widgets.Text(
            description='Full path:',
            disabled=True,
            layout={'width': '800px'},
            style={'description_width': 'initial'}
        )

        self.filepath_dropdown.observe(self._on_selection_change, names='value')
        self.path_display.value = self.filepath_dropdown.value

    def _on_selection_change(self, change):
        """Handle selection changes in the dropdown"""
        self.path_display.value = change['new']

    def display_selector(self):
        """Display the file selection widgets"""
        display(widgets.VBox([self.filepath_dropdown, self.path_display]))

    @property
    def selected_file(self):
        """Get the currently selected file path"""
        return self.filepath_dropdown.value