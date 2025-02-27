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
        files_dict = {
            "Bach | Cello Suite No. 1 in G Major": "./music_database/bach_cello_suites/Suite No. 1 in G major",
            "Bach | Cello Suite No. 2 in D Minor": "./music_database/bach_cello_suites/Suite No. 2 in D minor",
            "Bach | Cello Suite No. 3 in C Major": "./music_database/bach_cello_suites/Suite No. 3 in C major",
            "Bach | Cello Suite No. 4 in E-flat Major": "./music_database/bach_cello_suites/Suite No. 4 in Eb major",
            "Bach | Cello Suite No. 5 in C Minor": "./music_database/bach_cello_suites/Suite No. 5 in C minor",
            "Bach | Cello Suite No. 6 in D Major": "./music_database/bach_cello_suites/Suite No. 6 in D major",

            "Chopin | Étude Op. 10 No. 1 'Waterfall'": "./music_database/chopin_etude_op10/Chopin_-_Etude_Op.10_No.1_Waterfall_Chopin_in_C_Major.mxl",
            "Chopin | Étude Op. 10 No. 2 'Chromatic'": "./music_database/chopin_etude_op10/Frdric_Chopin_tude_in_A_minor_Op._10_No._2_Chromatic.mxl",
            "Chopin | Étude Op. 10 No. 4 'Torrent'": "./music_database/chopin_etude_op10/Frdric_Chopin_tude_in_C-sharp_minor_Op._10_No._4_Torrent.mxl",
            "Chopin | Étude Op. 10 No. 5 'Black Keys'": "./music_database/chopin_etude_op10/Frdric_Chopin_tude_in_G-flat_major_Op._10_No._5_Black_Keys.mxl",
            "Chopin | Étude Op. 10 No. 6 'Lament'": "./music_database/chopin_etude_op10/Frdric_Chopin_tude_in_E-flat_minor_Op._10_No._6_Lament.mxl",
            "Chopin | Étude Op. 10 No. 7 'Toccata'": "./music_database/chopin_etude_op10/Frdric_Chopin_tude_in_C_major_Op._10_No._7_Toccata.mxl",
            "Chopin | Étude Op. 10 No. 8 'Sunshine'": "./music_database/chopin_etude_op10/Frdric_Chopin_tude_in_F_major_Op._10_No._8_Sunshine.mxl",
            "Chopin | Étude Op. 10 No. 9": "./music_database/chopin_etude_op10/Frdric_Chopin_tude_in_F_minor_Op._10_No._9.mxl",
            "Chopin | Étude Op. 10 No. 10": "./music_database/chopin_etude_op10/Frdric_Chopin_tude_in_A-flat_major_Op._10_No._10.mxl",

            "Chopin | Étude Op. 25 No. 2 'The Bees'": "./music_database/chopin_etude_op25/Frdric_Chopin_tude__in_F_minor_Op._25_No._2_The_Bees.mxl",
            "Chopin | Étude Op. 25 No. 9 'Butterfly'": "./music_database/chopin_etude_op25/Frdric_Chopin_tude_in_G-flat_major_Op._25_No._9_Butterfly.mxl",
            "Chopin | Étude Op. 25 No. 11 'Winter Wind'": "./music_database/chopin_etude_op25/Chopin_-_Etude_Op.25_No.11_in_A_minor_Winter_Wind_-_F._Chopin.mxl",
            "Chopin | Étude Op. 25 No. 12 'Ocean'": "./music_database/chopin_etude_op25/Frdric_Chopin_tude_Op._25_No._12_in_C_minor_Ocean.mxl",

            "Chopin | Waltz Op. 34 No. 1 in A-flat Major": "./music_database/chopin_waltzes/Chopin_Waltz_in_A_Flat_Op._34_No._1.mxl",
            "Chopin | Waltz Op. 34 No. 2 in A Minor": "./music_database/chopin_waltzes/Waltz_in_A_Minor_Opus_34_No._2.mxl",
            "Chopin | Waltz Op. 34 No. 3 in F Major": "./music_database/chopin_waltzes/Waltz_in_F_major_Op.34_No.3.mxl",
            "Chopin | Waltz Op. 64 No. 1 'Minute Waltz'": "./music_database/chopin_waltzes/Chopin_-_Waltz_Opus_64_No._1_in_D_Major_Minute_Waltz.mxl",
            "Chopin | Waltz Op. 64 No. 2 in C Minor": "./music_database/chopin_waltzes/Chopin_-_Waltz_Opus_64_No._2_in_C_Minor.mxl",
            "Chopin | Waltz Op. 64 No. 3 in A-flat Major": "./music_database/chopin_waltzes/Waltzes_Op.64__Frdric_Chopin_Waltz_Op.64_No.3_in_A-flat_Major_-_Frederic_Chopin.mxl",
            "Chopin | Waltz Op. 42 in A Major": "./music_database/chopin_waltzes/Waltz_Opus_42_in_A_Major.mxl",

            "Mozart | Minuet in G Major": "./music_database/Mozart_-_Menuett_in_G_Major.mxl",
            "Mozart | Piano Sonata No. 10 in C Major, K. 330 (1st movement)": "./music_database/Mozart_-_Sonata_No._10_1st_Movement_K._330.mxl",
            "Mozart | Piano Sonata No. 11 in A Major, K. 331 'Rondo Alla Turca'": "./music_database/Mozart_-_Piano_Sonata_No._11_K._331_3rd_Movement_Rondo_alla_Turca.mxl",

            "Satie | Gnossienne No. 1": "./music_database/satie_gnossiennes/Satie_-_Gnossienne_No._1.mxl",
            "Satie | Gnossienne No. 2": "./music_database/satie_gnossiennes/Satie_-_Gnossienne_No._2.mxl",
            "Satie | Gnossienne No. 3": "./music_database/satie_gnossiennes/Satie_-_Gnossienne_No._3.mxl",
            "Satie | Gnossienne No. 4": "./music_database/satie_gnossiennes/Satie_-_Gnossienne_No._4.mxl",

            "Ysaÿe | Violin Sonata No. 1 in G Minor": "./music_database/ysaye_sonatas/Solo_Violin_Sonata_in_G_Minor_-_E._Ysae_Op._27_No._1.mxl",
            "Ysaÿe | Violin Sonata No. 2 in A Minor": "./music_database/ysaye_sonatas/Solo_Violin_Sonata_in_A_Minor_-_E._Ysae_Op._27_No._2.mxl",
            "Ysaÿe | Violin Sonata No. 3 in D Minor": "./music_database/ysaye_sonatas/Solo_Violin_Sonata_in_D_Minor_-_E._Ysae_Op._27_No._3.mxl",
            "Ysaÿe | Violin Sonata No. 4 in E Minor": "./music_database/ysaye_sonatas/Solo_Violin_Sonata_in_E_Minor_-_E._Ysae_Op._27_No._4.mxl",
            "Ysaÿe | Violin Sonata No. 5 in G Major": "./music_database/ysaye_sonatas/Solo_Violin_Sonata_in_G_Major_-_E._Ysae_Op._27_No._5.mxl",
            "Ysaÿe | Violin Sonata No. 6 in E Major": "./music_database/ysaye_sonatas/Solo_Violin_Sonata_in_E_Major_-_E._Ysae_Op._27_No._6.mxl"
        }

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
