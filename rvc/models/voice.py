import os

from dotenv import load_dotenv

load_dotenv()


class Voice:
    base_path = os.getenv("MODELS_PATH")
    specifications = {
        "kanye-west": {
            "title": "Kanye West",
            "files": {
                "pth": "kanye2333333.pth",
            },
        },
        "jordan-peterson": {
            "title": "Jordan Peterson",
            "files": {
                "pth": "JordanPeterson.pth",
            },
        },
    }

    def __init__(self, model):
        """Init"""
        from pathlib import Path

        root = Path(self.base_path).resolve()
        target = (root / model).resolve()
        if not target.is_relative_to(root) or target == root:
            raise ValueError("Model must be inside the configured model directory")
        self.model = model
        self.pth_file, self.index_file = self.find_pth_and_index_files(model)

    def find_pth_and_index_files(self, dir_path):
        """Find pth and index files in the directory path,
        handling missing specifications and files gracefully.
        """
        pth_file = None
        index_file = None

        if dir_path in self.specifications and "files" in self.specifications[dir_path]:
            if "pth" in self.specifications[dir_path]["files"]:
                pth_file = os.path.join(
                    self.base_path,
                    dir_path,
                    self.specifications[dir_path]["files"]["pth"],
                )
            if "index" in self.specifications[dir_path]["files"]:
                index_file = os.path.join(
                    self.base_path,
                    dir_path,
                    self.specifications[dir_path]["files"]["index"],
                )

        if not pth_file or not index_file:
            for root, _, files in os.walk(f"{self.base_path}/{dir_path}"):
                for file in files:
                    if (
                        not pth_file
                        and file.endswith(".pth")
                        and not file.startswith("._")
                    ):
                        pth_file = os.path.join(root, file)
                    elif (
                        not index_file
                        and file.endswith(".index")
                        and not file.startswith("._")
                    ):
                        index_file = os.path.join(root, file)

                if pth_file and index_file:
                    break

        if pth_file is None or index_file is None:
            raise FileNotFoundError(
                f"Could not find required files for model '{dir_path}' in '{self.base_path}'."
            )

        return pth_file, index_file
