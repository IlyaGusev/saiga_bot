import json
import sys
import shutil

from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Footer, MarkdownViewer, Static, Input, Button
from textual.containers import Horizontal


class IndexInput(Static):
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Input(placeholder="Enter index", id="index_input")
            yield Button("Go", variant="primary", id="go_button")


class RecordInfo(Static):
    def update_info(self, current: int, total: int):
        self.update(f"Record {current + 1} of {total}")


def to_markdown(record):
    result = ""
    messages = record["messages"]
    for m in messages:
        result += "# {role}\n{content}\n\n".format(role=m["role"], content=m["content"])
    return result


class Browser(App):
    CSS_PATH = "browser.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "delete", "Delete"),
        ("b", "back", "Back"),
        ("f", "forward", "Forward"),
        ("s", "save", "Save"),
    ]

    def compose(self) -> ComposeResult:
        self.path = sys.argv[1]
        self.current_idx = 0
        with open(sys.argv[1]) as r:
            self.records = [json.loads(line) for line in r]
        yield MarkdownViewer()
        yield RecordInfo()
        yield IndexInput()
        yield Footer()

    @property
    def markdown_viewer(self) -> MarkdownViewer:
        return self.query_one(MarkdownViewer)

    @property
    def footer(self) -> Footer:
        return self.query_one(Footer)

    @property
    def record_info(self) -> RecordInfo:
        return self.query_one(RecordInfo)

    @property
    def index_input(self) -> IndexInput:
        return self.query_one(IndexInput)

    async def show_record(self):
        if len(self.records) == 0:
            await self.markdown_viewer.document.update("No records left")
            self.record_info.update_info(-1, 0)
            return
        assert self.current_idx < len(self.records)
        await self.markdown_viewer.document.update(to_markdown(self.records[self.current_idx]))
        self.markdown_viewer.scroll_home(animate=False)
        self.record_info.update_info(self.current_idx, len(self.records))

    async def on_mount(self) -> None:
        await self.show_record()

    async def action_back(self) -> None:
        self.current_idx -= 1
        if self.current_idx < 0:
            self.current_idx = len(self.records) - 1
        await self.show_record()

    async def action_forward(self) -> None:
        self.current_idx += 1
        if self.current_idx >= len(self.records):
            self.current_idx = 0
        await self.show_record()

    async def action_delete(self) -> None:
        assert self.current_idx < len(self.records)
        current_item = self.records[self.current_idx]
        deleted_item = self.records.pop(self.current_idx)
        assert current_item == deleted_item
        self.current_idx += 1
        if self.current_idx >= len(self.records):
            self.current_idx = 0
        await self.show_record()

    def action_save(self) -> None:
        with open(self.path + "_tmp", "w") as w:
            for record in self.records:
                w.write(json.dumps(record, ensure_ascii=False) + "\n")
        shutil.move(self.path + "_tmp", self.path)

    @on(Button.Pressed, "#go_button")
    async def handle_go_button(self, event: Button.Pressed) -> None:
        await self.go_to_index()

    async def go_to_index(self) -> None:
        input_value = self.query_one("#index_input").value
        index = int(input_value) - 1
        if 0 <= index < len(self.records):
            self.current_idx = index
            await self.show_record()
        else:
            self.notify("Invalid index. Please enter a number between 1 and {}.".format(self.total_records))
        self.query_one("#index_input").value = ""


if __name__ == "__main__":
    Browser().run()
