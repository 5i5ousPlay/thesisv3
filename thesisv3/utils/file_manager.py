class MusicFileManager:
    """Handles music file selection and management through a UI interface"""

    def __init__(self, base_path="../Music Database"):
        self.base_path = Path(base_path)
        self.files = self._initialize_file_mapping()
        self.filepath_dropdown = None
        self.path_display = None
        self._setup_widgets()

    def _initialize_file_mapping(self):
        """Initialize the mapping of display names to file paths"""
        # return {
        #     "Liszt - Ungarische Rhapsodie": self.base_path / "GTTM Database/Franz Liszt/Ungarische Rhapsodie S.244 Nr.2 cis moll.xml",
        #     "Liszt - Liebestraume": self.base_path / "GTTM Database/Franz Liszt/Liebestraume 3 Notturnos S.541 R.211 As dur.xml",
        #     "Tchaikovsky - Swan Lake Finale": self.base_path / "GTTM Database/Pyotr Il'yich Tchaikovsky/Swan Lake Op.20 No.9 Finale.xml",
        #     # Add other files...
        # }
        return {
            "Liszt - Ungarische Rhapsodie": r"..\Music Database\GTTM Database\Franz Liszt\Ungarische Rhapsodie S.244 Nr.2 cis moll.xml",
            "Liszt - Liebestraume": r"..\Music Database\GTTM Database\Franz Liszt\Liebestraume 3 Notturnos S.541 R.211 As dur.xml",
            "Tchaikovsky - Swan Lake Finale": r"..\Music Database\GTTM Database\Pyotr Il’yich Tchaikovsky\Swan Lake Op.20 No.9 Finale.xml",
            "Segment 1": r"..\Music Database\fabricated\segment1.mid",
            "Segment 2": r"..\Music Database\fabricated\segment2.mid",
            "Segment 3": r"..\Music Database\fabricated\segment3.mid",
            "Segment 4": r"..\Music Database\fabricated\segment4.mid",
            "Segment 5": r"..\Music Database\fabricated\segment5.mid",
            "Fabricated": r"..\Music Database\fabricated\fabricated.mxl",
            "Fabricated2": r"..\Music Database\fabricated\fabricated2.mxl",
            "Mountain King": r"..\\In_the_Hall_of_the_Mountain_King_Easy_variation2.mxl",
            "Bach - Minuet in G Major": r"..\Music Database\Good_maybe\Bach_-_Minuet_in_G_Major_Bach.mxl"
        }

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