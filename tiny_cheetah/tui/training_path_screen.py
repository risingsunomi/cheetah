from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from typing import List, Optional

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.screen import Screen, ModalScreen
from textual.widget import Widget
from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView, Static

from tiny_cheetah.tui.training_path_types import TrainingNode, NODE_STATUS_STYLES, NODE_STATUS_SYMBOLS


class TrainingPathScreen(Screen[Optional[List[TrainingNode]]]):
    """Dedicated editor for the sequential training path."""

    CSS_PATH = Path(__file__).with_name("training_path.tcss")

    BINDINGS = [
        Binding("escape", "cancel", "Back"),
    ]

    def __init__(self, nodes: List[TrainingNode]) -> None:
        super().__init__()
        self._path_nodes: List[TrainingNode] = copy.deepcopy(nodes) or [TrainingNode("Base Training")]
        self._ensure_base_step()
        self._path_list: Optional[ListView] = None
        self._graph_canvas: Optional[Container] = None
        self._feedback: Optional[Label] = None
        self._mode_label: Optional[Label] = None
        self._selected_index = 0
        self._drag_index: Optional[int] = None
        self._rename_button: Optional[Button] = None
        self._delete_button: Optional[Button] = None
        self._step_settings_button: Optional[Button] = None
        self._scratch_button: Optional[Button] = None
        self._finetune_button: Optional[Button] = None

    def compose(self) -> ComposeResult:
        items = [self._build_list_item(index, node) for index, node in enumerate(self._path_nodes)]
        yield Header(show_clock=True)
        with Container(id="path-root"):
            yield Label("Training Path Editor", id="path-title")
            with Container(id="path-body"):
                with VerticalScroll(id="path-graph-scroll"):
                    graph = Container(id="path-graph")
                    self._graph_canvas = graph
                    yield graph
                with Container(id="path-sidebar"):
                    list_view = ListView(*items, id="path-list")
                    self._path_list = list_view
                    yield list_view
                    feedback = Label("", id="path-feedback")
                    self._feedback = feedback
                    yield feedback
                    mode_label = Label("", id="path-mode-label")
                    self._mode_label = mode_label
                    yield mode_label
                    with Container(id="path-actions"):
                        with Container(classes="path-action-row"):
                            yield Button("Add Step", id="path-add", variant="primary")
                            step_settings_btn = Button("Step Settings", id="path-step-settings")
                            self._step_settings_button = step_settings_btn
                            yield step_settings_btn
                        with Container(classes="path-action-row"):
                            rename_btn = Button("Rename Step", id="path-rename")
                            self._rename_button = rename_btn
                            yield rename_btn
                            delete_btn = Button("Delete Step", id="path-delete", variant="error")
                            self._delete_button = delete_btn
                            yield delete_btn
                        with Container(classes="path-action-row"):
                            scratch_btn = Button("Set Scratch", id="path-mode-scratch")
                            self._scratch_button = scratch_btn
                            yield scratch_btn
                            finetune_btn = Button("Set Fine-tune", id="path-mode-finetune")
                            self._finetune_button = finetune_btn
                            yield finetune_btn
                        with Container(classes="path-action-row path-action-footer"):
                            yield Button("Reset", id="path-reset", variant="warning")
                            yield Button("Cancel", id="path-cancel")
                            yield Button("Save & Return", id="path-save", variant="success")
        yield Footer()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_mount(self) -> None:
        self._refresh_views()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        handlers = {
            "path-add": self._handle_add,
            "path-step-settings": lambda: self._handle_step_settings(self._selected_index),
            "path-rename": lambda: self._handle_rename(self._selected_index),
            "path-delete": lambda: self._handle_delete(self._selected_index),
            "path-mode-scratch": lambda: self._set_selected_mode(True),
            "path-mode-finetune": lambda: self._set_selected_mode(False),
            "path-reset": self._handle_reset,
            "path-cancel": lambda: self.dismiss(None),
            "path-save": self._handle_save,
        }
        handler = handlers.get(event.button.id)
        if handler is not None:
            handler()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        index = self._index_from_item(event.item)
        if index is not None:
            self._selected_index = index
            self._refresh_graph()
            self._update_action_buttons()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = self._index_from_item(event.item)
        if index is None:
            return
        self._selected_index = index
        self._refresh_graph()
        self._update_action_buttons()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if event.button == 1:
            node_index = self._graph_node_index_from_control(event.control)
            if node_index is not None:
                self._drag_index = node_index

    def on_mouse_up(self, event: events.MouseUp) -> None:
        if self._drag_index is None:
            return
        target_index = self._graph_node_index_from_control(event.control)
        if target_index is not None and target_index != self._drag_index:
            self._move_node(self._drag_index, target_index)
        self._drag_index = None

    # ---- Event handlers -------------------------------------------------

    def _handle_add(self) -> None:
        modal = NodeStepModal()
        self.app.push_screen(modal, self._on_add_result)

    def _handle_rename(self, index: int) -> None:
        if not self._path_nodes:
            self._set_feedback("No steps to rename.")
            return
        if index <= 0:
            self._set_feedback("Base step name is derived from the model.")
            return
        current = self._path_nodes[index]
        modal = NodeStepModal(title="Rename Training Step", initial=current.name, confirm_label="Save")
        self.app.push_screen(modal, lambda result: self._on_rename_result(index, result))

    def _handle_delete(self, index: int) -> None:
        if len(self._path_nodes) <= 1 or index <= 0:
            self._set_feedback("Cannot delete the base step.")
            return
        node = self._path_nodes[index]
        modal = PathConfirmModal(
            f"Delete '{node.name}'?",
            "This step will be removed permanently.",
            confirm_label="Delete",
            confirm_variant="error",
        )
        self.app.push_screen(modal, lambda confirmed: self._on_delete_result(index, confirmed))

    def _handle_step_settings(self, index: int) -> None:
        if not self._path_nodes:
            self._set_feedback("No steps available.")
            return
        if index <= 0:
            self._set_feedback("Base step uses the main Train Settings screen.")
            return
        current = self._path_nodes[index]
        modal = PathStepSettingsModal(current.settings)
        self.app.push_screen(modal, lambda result: self._on_step_settings_result(index, result))

    def _handle_reset(self) -> None:
        modal = PathConfirmModal(
            "Reset Training Path?",
            "This restores the path to a single base training step.",
            confirm_label="Reset",
            confirm_variant="warning",
        )
        self.app.push_screen(modal, self._on_reset_confirmed)

    def _handle_save(self) -> None:
        for node in self._path_nodes:
            node.status = "pending"
        self._ensure_base_step()
        self.dismiss(copy.deepcopy(self._path_nodes))

    # ---- Results from modals -------------------------------------------

    def _on_add_result(self, result: Optional[str]) -> None:
        if result is None:
            return
        name = result.strip()
        if not name:
            self._set_feedback("Step name cannot be blank.")
            return
        base_settings = copy.deepcopy(self._path_nodes[0].settings)
        base_settings["from-scratch"] = False
        self._path_nodes.append(TrainingNode(name, settings=base_settings))
        self._selected_index = len(self._path_nodes) - 1
        self._refresh_views()
        self._set_feedback(f"Added step '{name}'.")

    def _on_rename_result(self, index: int, result: Optional[str]) -> None:
        if result is None:
            return
        name = result.strip()
        if not name:
            self._set_feedback("Step name cannot be blank.")
            return
        self._path_nodes[index].name = name
        self._refresh_views()
        self._set_feedback(f"Step renamed to '{name}'.")

    def _on_delete_result(self, index: int, confirmed: Optional[bool]) -> None:
        if not confirmed:
            return
        removed = self._path_nodes.pop(index)
        if self._selected_index >= len(self._path_nodes):
            self._selected_index = len(self._path_nodes) - 1
        self._refresh_views()
        self._set_feedback(f"Removed step '{removed.name}'.")

    def _on_step_settings_result(self, index: int, result: Optional[dict[str, str]]) -> None:
        if result is None:
            return
        node = self._path_nodes[index]
        preserved_mode = bool(node.settings.get("from-scratch", False))
        updated = dict(node.settings)
        for key, value in result.items():
            if key in {"dataset-id", "data-path"}:
                continue
            cleaned = value.strip()
            if cleaned:
                updated[key] = cleaned
            else:
                updated.pop(key, None)
        dataset = result.get("dataset-id", "").strip()
        data_path = result.get("data-path", "").strip()
        if dataset:
            updated["dataset-id"] = dataset
            updated["data-path"] = ""
        elif data_path:
            updated["data-path"] = data_path
            updated["dataset-id"] = ""
        else:
            updated.pop("dataset-id", None)
            updated.pop("data-path", None)
        updated["from-scratch"] = preserved_mode
        node.settings = updated
        self._refresh_views()
        self._set_feedback(f"Updated settings for '{node.name}'.")

    def _on_reset_confirmed(self, confirmed: Optional[bool]) -> None:
        if not confirmed:
            return
        self._path_nodes = [TrainingNode("Base Training")]
        self._selected_index = 0
        self._refresh_views()
        self._set_feedback("Training path reset to base step.")

    # ---- Rendering ------------------------------------------------------

    def _refresh_views(self) -> None:
        asyncio.create_task(self._async_refresh_views())

    async def _async_refresh_views(self) -> None:
        await self._refresh_list()
        await self._refresh_graph()
        self._update_action_buttons()

    async def _refresh_list(self) -> None:
        if self._path_list is None:
            return
        await self._path_list.clear()
        for index, node in enumerate(self._path_nodes):
            await self._path_list.mount(self._build_list_item(index, node))
        if self._path_nodes:
            target = max(0, min(self._selected_index, len(self._path_nodes) - 1))
            try:
                self._path_list.index = target
            except AttributeError:
                pass

    async def _refresh_graph(self) -> None:
        if self._graph_canvas is None:
            return
        await self._graph_canvas.remove_children()
        if not self._path_nodes:
            await self._graph_canvas.mount(Static("No steps defined", classes="graph-empty"))
            return
        for index, node in enumerate(self._path_nodes):
            node_box = Static(
                self._format_node_label(index, node),
                classes=self._graph_classes(index, node),
                markup=True,
                id=f"graph-node-{index}",
            )
            await self._graph_canvas.mount(node_box)
            if index < len(self._path_nodes) - 1:
                connector = Static("│\n│\n▼", classes="graph-connector", markup=False)
                await self._graph_canvas.mount(connector)

    # ---- Helpers --------------------------------------------------------

    def _build_list_item(self, index: int, node: TrainingNode) -> ListItem:
        label = Label(self._format_list_label(index, node), markup=True)
        return ListItem(label, id=f"path-step-{index}")

    def _format_list_label(self, index: int, node: TrainingNode) -> str:
        symbol = NODE_STATUS_SYMBOLS.get(node.status, "•")
        color = NODE_STATUS_STYLES.get(node.status, "#bbbbbb")
        repeated = " (repeat)" if node.repeated else ""
        return (
            f"[{color}]{symbol}[/{color}] Step {index + 1}: "
            f"{node.name}{repeated} [dim]· {self._mode_text(index, node)}[/]"
        )

    def _format_node_label(self, index: int, node: TrainingNode) -> str:
        symbol = NODE_STATUS_SYMBOLS.get(node.status, "•")
        status = node.status.capitalize()
        repeated = "Repeat" if node.repeated else "Sequential"
        return f"[bold]{symbol} {node.name}[/]\n[dim]{status} · {repeated} · {self._mode_text(index, node)}[/]"

    def _graph_classes(self, index: int, node: TrainingNode) -> str:
        classes = ["graph-node", f"status-{node.status}"]
        if node.repeated:
            classes.append("repeat")
        if index == self._selected_index:
            classes.append("selected")
        return " ".join(classes)

    def _ensure_base_step(self) -> None:
        if not self._path_nodes:
            self._path_nodes.append(TrainingNode("Base Training"))
        else:
            self._path_nodes[0].name = self._path_nodes[0].name or "Base Training"
        self._path_nodes[0].status = (
            self._path_nodes[0].status if self._path_nodes[0].status in NODE_STATUS_STYLES else "pending"
        )
        self._path_nodes[0].repeated = False
        self._path_nodes[0].settings.setdefault("from-scratch", False)

    def _selected_node(self) -> Optional[TrainingNode]:
        if 0 <= self._selected_index < len(self._path_nodes):
            return self._path_nodes[self._selected_index]
        return None

    def _mode_text(self, index: int, node: TrainingNode) -> str:
        if index == 0:
            return "mode via Train Settings"
        return "from scratch" if bool(node.settings.get("from-scratch", False)) else "fine-tune previous"

    def _step_settings_text(self, index: int, node: TrainingNode) -> str:
        if index == 0:
            return "data and tuning come from the main Train Settings screen"
        dataset = str(node.settings.get("dataset-id", "")).strip()
        data_path = str(node.settings.get("data-path", "")).strip()
        limit = str(node.settings.get("max-dataset-entries", "")).strip()
        epochs = str(node.settings.get("epochs", "")).strip()
        lr = str(node.settings.get("lr", "")).strip()
        parts: List[str] = []
        if dataset:
            parts.append(f"dataset={dataset}")
        elif data_path:
            parts.append(f"data={data_path}")
        else:
            parts.append("data inherits main settings")
        if limit:
            parts.append(f"max_entries={limit}")
        if epochs:
            parts.append(f"epochs={epochs}")
        if lr:
            parts.append(f"lr={lr}")
        return " | ".join(parts)

    def _index_from_item(self, item: Optional[ListItem]) -> Optional[int]:
        if item is None or item.id is None:
            return None
        try:
            return int(item.id.split("-")[-1])
        except (ValueError, IndexError):
            return None

    def _set_feedback(self, message: str) -> None:
        if self._feedback is not None:
            self._feedback.update(message)

    def _graph_node_index_from_control(self, control: Optional[Widget]) -> Optional[int]:
        target = control
        while target is not None:
            if target.id and target.id.startswith("graph-node-"):
                try:
                    return int(target.id.split("-", 2)[2])
                except (IndexError, ValueError):
                    return None
            target = target.parent
        return None

    def _move_node(self, source: int, target: int) -> None:
        if source == target or target < 0 or target >= len(self._path_nodes):
            return
        if source == 0 or target == 0:
            self._set_feedback("Base step remains at the top.")
            return
        node = self._path_nodes.pop(source)
        target = max(0, min(target, len(self._path_nodes)))
        self._path_nodes.insert(target, node)
        self._selected_index = target
        self._refresh_views()
        self._set_feedback(f"Moved step to position {target + 1}.")

    def _set_selected_mode(self, from_scratch: bool) -> None:
        node = self._selected_node()
        if node is None:
            self._set_feedback("No step selected.")
            return
        if self._selected_index == 0:
            self._set_feedback("Base step mode is controlled from Train Settings.")
            return
        node.settings["from-scratch"] = from_scratch
        self._refresh_views()
        self._set_feedback(
            f"Step '{node.name}' set to {'from scratch' if from_scratch else 'fine-tune previous'}."
        )

    def _update_action_buttons(self) -> None:
        allow_rename = self._selected_index > 0
        allow_delete = self._selected_index > 0 and len(self._path_nodes) > 1
        if self._rename_button is not None:
            self._rename_button.disabled = not allow_rename
        if self._delete_button is not None:
            self._delete_button.disabled = not allow_delete
        if self._step_settings_button is not None:
            self._step_settings_button.disabled = self._selected_index == 0
        if self._scratch_button is not None:
            self._scratch_button.disabled = self._selected_index == 0
        if self._finetune_button is not None:
            self._finetune_button.disabled = self._selected_index == 0
        if self._mode_label is not None:
            node = self._selected_node()
            if node is None:
                self._mode_label.update("Mode: --")
            else:
                self._mode_label.update(
                    f"Mode: {self._mode_text(self._selected_index, node)}\n"
                    f"Data: {self._step_settings_text(self._selected_index, node)}"
                )

class NodeStepModal(ModalScreen[Optional[str]]):
    """Modal dialog for capturing a training step name."""

    BINDINGS = [Binding("escape", "dismiss", "Cancel")]

    def __init__(
        self,
        *,
        title: str = "Add Training Step",
        initial: str = "",
        confirm_label: str = "Add",
        placeholder: str = "e.g. Fine-tune dataset",
    ) -> None:
        super().__init__(id="node-step-modal")
        self._title = title
        self._initial = initial
        self._confirm_label = confirm_label
        self._placeholder = placeholder
        self._name_input: Optional[Input] = None

    def compose(self) -> ComposeResult:
        with Container(id="node-step-modal-container"):
            yield Label(self._title, id="node-step-title")
            self._name_input = Input(placeholder=self._placeholder, id="node-step-name")
            yield self._name_input
            with Container(id="node-step-buttons"):
                yield Button("Cancel", id="node-step-cancel")
                yield Button(self._confirm_label, id="node-step-confirm", variant="primary")

    def on_mount(self) -> None:
        if self._name_input is not None:
            self._name_input.value = self._initial
            self.call_after_refresh(self._name_input.focus)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "node-step-cancel":
            self.dismiss(None)
        elif event.button.id == "node-step-confirm":
            value = self._name_input.value.strip() if self._name_input else ""
            self.dismiss(value or None)


class PathStepSettingsModal(ModalScreen[Optional[dict[str, str]]]):
    """Modal for editing per-step data and tuning overrides."""

    BINDINGS = [Binding("escape", "dismiss", "Cancel")]

    _FIELDS = [
        ("dataset-id", "Dataset ID", "Optional HF dataset for this step"),
        ("data-path", "Data Path", "Optional local UTF-8 corpus for this step"),
        ("max-dataset-entries", "Max Entries", "Optional dataset entry cap"),
        ("epochs", "Epochs", "Leave blank to inherit"),
        ("lr", "Learning Rate", "Leave blank to inherit"),
        ("seq-length", "Seq Length", "Leave blank to inherit"),
        ("batch-size", "Batch Size", "Leave blank to inherit"),
        ("gradient-accumulation", "Grad Accum", "Leave blank to inherit"),
    ]

    def __init__(self, values: dict[str, object]) -> None:
        super().__init__(id="path-step-settings-modal")
        self._initial = dict(values)
        self._inputs: dict[str, Input] = {}

    def compose(self) -> ComposeResult:
        with Container(id="path-step-settings-modal-container"):
            yield Label("Step Settings", id="path-step-settings-title")
            yield Static(
                "Leave fields blank to inherit Train Settings. "
                "Set either Dataset ID or Data Path when this step should fine-tune on a different corpus.",
                id="path-step-settings-help",
            )
            with VerticalScroll(id="path-step-settings-scroll"):
                for name, label, placeholder in self._FIELDS:
                    yield Label(label, classes="path-step-settings-label")
                    widget = Input(id=f"path-step-settings-{name}", placeholder=placeholder)
                    widget.add_class("path-step-settings-input")
                    self._inputs[name] = widget
                    yield widget
            with Container(id="path-step-settings-buttons"):
                yield Button("Cancel", id="path-step-settings-cancel")
                yield Button("Apply", id="path-step-settings-apply", variant="primary")

    def on_mount(self) -> None:
        for name, widget in self._inputs.items():
            value = self._initial.get(name)
            if value is not None:
                widget.value = str(value).strip()
        first = self._inputs.get("dataset-id")
        if first is not None:
            self.call_after_refresh(first.focus)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "path-step-settings-cancel":
            self.dismiss(None)
        elif event.button.id == "path-step-settings-apply":
            self.dismiss(self._gather_values())

    def _gather_values(self) -> dict[str, str]:
        return {
            name: widget.value.strip()
            for name, widget in self._inputs.items()
        }


class PathConfirmModal(ModalScreen[Optional[bool]]):
    """Confirmation modal for destructive operations."""

    BINDINGS = [Binding("escape", "dismiss", "Cancel")]

    def __init__(
        self,
        title: str,
        message: str,
        *,
        confirm_label: str = "Confirm",
        confirm_variant: str = "primary",
    ) -> None:
        super().__init__(id="path-confirm-modal")
        self._title = title
        self._message = message
        self._confirm_label = confirm_label
        self._confirm_variant = confirm_variant

    def compose(self) -> ComposeResult:
        with Container(id="path-confirm-modal-container"):
            yield Label(self._title, id="path-confirm-title")
            yield Static(self._message, id="path-confirm-message")
            with Container(id="path-confirm-buttons"):
                yield Button("Cancel", id="path-confirm-cancel")
                yield Button(self._confirm_label, id="path-confirm-accept", variant=self._confirm_variant)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "path-confirm-cancel":
            self.dismiss(False)
        elif event.button.id == "path-confirm-accept":
            self.dismiss(True)
