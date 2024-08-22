from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union
from enum import Enum

class TextColor(Enum):
    BLACK = "black"
    DARK_BLUE = "dark_blue"
    DARK_GREEN = "dark_green"
    DARK_AQUA = "dark_aqua"
    DARK_RED = "dark_red"
    DARK_PURPLE = "dark_purple"
    GOLD = "gold"
    GRAY = "gray"
    DARK_GRAY = "dark_gray"
    BLUE = "blue"
    GREEN = "green"
    AQUA = "aqua"
    RED = "red"
    LIGHT_PURPLE = "light_purple"
    YELLOW = "yellow"
    WHITE = "white"
    RESET = "reset"

class ClickEventAction(Enum):
    OPEN_URL = "open_url"
    OPEN_FILE = "open_file"
    RUN_COMMAND = "run_command"
    SUGGEST_COMMAND = "suggest_command"
    CHANGE_PAGE = "change_page"
    COPY_TO_CLIPBOARD = "copy_to_clipboard"

class HoverEventAction(Enum):
    SHOW_TEXT = "show_text"
    SHOW_ITEM = "show_item"
    SHOW_ENTITY = "show_entity"

@dataclass
class ClickEvent:
    action: ClickEventAction
    value: str

@dataclass
class HoverEventContents:
    id: Optional[str] = None
    count: Optional[int] = None
    tag: Optional[str] = None
    name: Optional[RawJson] = None
    type: Optional[str] = None
    uuid: Optional[str] = None

@dataclass
class HoverEvent:
    action: HoverEventAction
    contents: Union[RawJson, HoverEventContents]

@dataclass
class RawJson:
    text: Optional[str] = None
    translate: Optional[str] = None
    with_: Optional[List[RawJson]] = field(default_factory=list)
    score: Optional[Dict[str, str]] = None
    selector: Optional[str] = None
    keybind: Optional[str] = None
    nbt: Optional[str] = None
    type: Optional[str] = None
    extra: Optional[List[RawJson]] = None
    color: Optional[Union[TextColor, str]] = None
    font: Optional[str] = None
    bold: Optional[bool] = None
    italic: Optional[bool] = None
    underlined: Optional[bool] = None
    strikethrough: Optional[bool] = None
    obfuscated: Optional[bool] = None
    insertion: Optional[str] = None
    click_event: Optional[ClickEvent] = None
    hover_event: Optional[HoverEvent] = None

    def __post_init__(self):
        if self.text is not None:
            self.type = "text"
        elif self.with_:
            self.type = "translatable"
        elif self.nbt is not None:
            self.type = "nbt"
        elif self.selector is not None:
            self.type = "selector"
        elif self.keybind is not None:
            self.type = "keybind"
        elif self.score is not None:
            self.type = "score"

    def __str__(self):
        import json
        from Helper.helpers import minecraft_command_escape
        return minecraft_command_escape(json.dumps(self.to_dict()))

    def to_dict(self) -> Dict:
        result = {}
        for key, value in {k: v for k, v in vars(self).items() if not k.startswith('__') and not callable(v)}.items():
            if value is not None:
                if key == "with_":
                    if len(value) > 0:
                        result["with"] = [item.to_dict() for item in value]
                elif key == "color" and isinstance(value, TextColor):
                    result[key] = value.value
                elif isinstance(value, (ClickEvent, HoverEvent)):
                    result[key] = str(value.__dict__)
                elif isinstance(value, list) and all(isinstance(item, RawJson) for item in value):
                    result[key] = [item.to_dict() for item in value if isinstance(item, RawJson)]
                else:
                    result[key] = str(value)
        return result

    @classmethod
    def text(cls, text: str, **kwargs) -> RawJson:
        return cls(text=text, **kwargs)

    @classmethod
    def translate(cls, key: str, with_: List[RawJson] = None, **kwargs) -> RawJson:
        return cls(translate=key, with_=with_, **kwargs)

    @classmethod
    def score(cls, name: str, objective: str, **kwargs) -> RawJson:
        return cls(score={"name": name, "objective": objective}, **kwargs)

    @classmethod
    def selector(cls, selector: str, **kwargs) -> RawJson:
        return cls(selector=selector, **kwargs)

    @classmethod
    def keybind(cls, key: str, **kwargs) -> RawJson:
        return cls(keybind=key, **kwargs)

    @classmethod
    def nbt(cls, path: str, **kwargs) -> RawJson:
        return cls(nbt=path, **kwargs)

    def with_color(self, color: Union[TextColor, str]) -> RawJson:
        self.color = color
        return self

    def with_formatting(self, bold: bool = None, italic: bool = None, underlined: bool = None,
                        strikethrough: bool = None, obfuscated: bool = None) -> RawJson:
        self.bold = bold if bold is not None else self.bold
        self.italic = italic if italic is not None else self.italic
        self.underlined = underlined if underlined is not None else self.underlined
        self.strikethrough = strikethrough if strikethrough is not None else self.strikethrough
        self.obfuscated = obfuscated if obfuscated is not None else self.obfuscated
        return self

    def with_click_event(self, action: ClickEventAction, value: str) -> RawJson:
        self.click_event = ClickEvent(action, value)
        return self

    def with_hover_event(self, action: HoverEventAction, contents: Union[RawJson, HoverEventContents]) -> RawJson:
        self.hover_event = HoverEvent(action, contents)
        return self

    def add_extra(self, *extra: RawJson) -> RawJson:
        if self.extra is None:
            self.extra = []
        self.extra.extend(extra)
        return self