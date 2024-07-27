import json
import sys
import shutil

from textual import on, events
from textual.app import App, ComposeResult, Binding
from textual.widgets import Footer, MarkdownViewer, Static, Input
from textual.validation import Number
from textual.containers import Container


class RecordInfo(Static):
    def update_info(self, current: int, total: int):
        self.update(f"Record {current + 1} of {total}")


class FullScreenLoadingIndicator(Static):
    def render(self) -> str:
        return "Loading..."


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
        Binding("g", "go", "Go", show=False, priority=True),
    ]

    def compose(self) -> ComposeResult:
        self.path = sys.argv[1]
        self.current_idx = 0
        with open(sys.argv[1]) as r:
            self.records = [json.loads(line) for line in r]
        yield Container(
            MarkdownViewer(),
            FullScreenLoadingIndicator(),
            id="main-content"
        )
        yield RecordInfo()
        yield Input(placeholder="Enter index", validators=[Number()], restrict="[0-9]*", valid_empty=True)
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
    def input(self) -> Input:
        return self.query_one(Input)

    @property
    def loading_indicator(self) -> FullScreenLoadingIndicator:
        return self.query_one(FullScreenLoadingIndicator)

    async def show_record(self):
        if len(self.records) == 0:
            await self.markdown_viewer.document.update("No records left")
            self.record_info.update_info(-1, 0)
            return

        assert self.current_idx < len(self.records)

        self.markdown_viewer.display = False
        self.loading_indicator.display = True

        await self.markdown_viewer.document.update(to_markdown(self.records[self.current_idx]))
        self.record_info.update_info(self.current_idx, len(self.records))

        def show_markdown():
            self.markdown_viewer.focus()
            self.markdown_viewer.display = True
            self.loading_indicator.display = False

        self.markdown_viewer.scroll_home(animate=False, on_complete=show_markdown)

    async def on_mount(self) -> None:
        self.loading_indicator.display = False
        await self.show_record()

    @on(Input.Submitted)
    async def goto(self, event: Input.Submitted) -> None:
        if not event.validation_result.is_valid:
            self.notify("Invalid index. Please enter a number between 1 and {}.".format(len(self.records)))
            return

        input_value = self.input.value
        index = int(input_value) - 1
        if 0 <= index < len(self.records):
            self.current_idx = index
            await self.show_record()
        else:
            self.notify("Invalid index. Please enter a number between 1 and {}.".format(len(self.records)))
        self.input.clear()

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
        assert 0 <= self.current_idx < len(self.records)
        self.records.pop(self.current_idx)
        self.current_idx += 1
        if self.current_idx >= len(self.records):
            self.current_idx = 0
        await self.show_record()

    async def action_go(self) -> None:
        if self.input.has_focus:
            validation_result = self.input.validate(self.input.value)
            self.post_message(self.input.Submitted(self.input, self.input.value, validation_result))

    def on_key(self, event: events.Key) -> None:
        if event.key in "1234567890" and not self.input.has_focus:
            self.input.focus()
            self.input.value = event.key

    def action_save(self) -> None:
        with open(self.path + "_tmp", "w") as w:
            for record in self.records:
                w.write(json.dumps(record, ensure_ascii=False) + "\n")
        shutil.move(self.path + "_tmp", self.path)


if __name__ == "__main__":
    Browser().run()
