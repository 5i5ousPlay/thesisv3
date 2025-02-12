from pathlib import Path

import ipywidgets as widgets
from IPython.display import display


class MusicFileManager:
    """Handles music file selection and management through a UI interface"""

    def __init__(self, base_path="./music_database"):
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
            "Bach - Minuet in G Major": r".\music_database\Bach_-_Minuet_in_G_Major_Bach.mxl",
            "Beethoven - Piano Sonata No. 20 in G Major Op. 49 1st Movement": r".\music_database\Beethoven_-_Piano_Sonata_No._20_in_G_Major_Op._49_1st_Movement.mxl",
            "Chopin - Ballade No.1 in G minor Op. 23 violin": r".\music_database\Chopin_-_Ballade_No.1_in_G_minor_Op._23_violin.mxl",
            "Chopin - Etude Op.10 No.1 Waterfall in C Major": r".\music_database\Chopin_-_Etude_Op.10_No.1_Waterfall_Chopin_in_C_Major.mxl",
            "Chopin - Etude Op.25 No.11 in A minor Winter Wind": r".\music_database\Chopin_-_Etude_Op.25_No.11_in_A_minor_Winter_Wind_-_F._Chopin.mxl",
            "Chopin - Nocturne Op.9 No.2 in E Flat Major": r".\music_database\Chopin_-_Nocturne_Op.9_No.2_E_Flat_Major.mxl",
            "Chopin - Waltz Opus 64 No.1 in D Major Minute Waltz": r".\music_database\Chopin_-_Waltz_Opus_64_No._1_in_D_Major_Minute_Waltz.mxl",
            "Chopin - Waltz Opus 64 No.2 in C Minor": r".\music_database\Chopin_-_Waltz_Opus_64_No._2_in_C_Minor.mxl",
            "Chopin - Waltz Opus 69 No.1 in A Major": r".\music_database\Chopin_-_Waltz_Opus_69_No._1_in_A_Major.mxl",
            "Mozart - Menuett in G Major": r".\music_database\Mozart_-_Menuett_in_G_Major.mxl",
            "Mozart - Piano Sonata No.11 K.331 3rd Movement Rondo alla Turca": r".\music_database\Mozart_-_Piano_Sonata_No._11_K_331_3rd_Movement_Rondo_alla_Turca.mxl",
            "Mozart - Sonata No.10 1st Movement K.330": r".\music_database\Mozart_-_Sonata_No._10_1st_Movement_K._330.mxl",
            "Satie - Gnossienne No.1": r".\music_database\Satie_-_Gnossienne_No._1.mxl"
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