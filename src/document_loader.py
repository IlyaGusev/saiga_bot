import io
import csv
import traceback
from typing import cast, Optional, BinaryIO

from pdfminer.high_level import extract_text as pdf_extract_text


class DocumentLoader:
    def __init__(self) -> None:
        self.parsers = {".csv": self.parse_csv, ".txt": self.parse_txt, ".pdf": self.parse_pdf}

    def load(self, stream: BinaryIO, file_ext: str) -> Optional[str]:
        handler = self.parsers.get(file_ext)
        if handler:
            try:
                return handler(stream)
            except Exception:
                traceback.print_exc()
        return None

    def is_supported(self, file_ext: str) -> bool:
        return file_ext in self.parsers

    def parse_csv(self, stream: BinaryIO) -> Optional[str]:
        wrapper = io.TextIOWrapper(stream, encoding="utf-8")
        csv_reader = csv.DictReader(wrapper)
        records = []
        for i, row in enumerate(csv_reader):
            text_row = []
            for k, v in row.items():
                key = k.strip() if isinstance(k, str) else k
                value = v.strip() if isinstance(v, str) else v
                text_row.append(f"{key}: {value}")
            records.append("\n".join(text_row))
        return "\n\n".join(records)

    def parse_txt(self, stream: BinaryIO) -> Optional[str]:
        wrapper = io.TextIOWrapper(stream, encoding="utf-8")
        return wrapper.read()

    def parse_pdf(self, stream: BinaryIO) -> Optional[str]:
        return pdf_extract_text(cast(io.IOBase, stream))
