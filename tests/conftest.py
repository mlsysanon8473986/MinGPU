import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mingpu import BESIROGLU, PairProgress, SearchSpace, run_outer_search  # noqa: E402
from mingpu.storage.db import connect, migrate  # noqa: E402

#: The small search the tests share: l = 8 and 16 on 8 and 16 H100 GPUs, t = 8, no TP overlap.
#: Its four pairs keep 18 to 132 parallel plans each, about 15 s of Calculon in all.
SMALL_SEARCH = dict(
    target_loss=2.6, layers=(8, 16), n_gpus=(8, 16), seq_len=2048, b_max=32, vocab=128_256,
    gpu="h100", datatype="bfloat16", law=BESIROGLU, top_k=1, no_tp_overlap=True, tensor_par=8,
)  # fmt: skip


@dataclass(frozen=True)
class SmallSearch:
    path: Path
    space: SearchSpace
    search_id: str
    events: list[PairProgress]

    def copy_to(self, directory: Path) -> "SmallSearch":
        """The same search in a database file of its own, so a test cannot change another's."""
        path = directory / self.path.name
        shutil.copyfile(self.path, path)
        return replace(self, path=path)


@pytest.fixture(scope="session")
def small_search_template(tmp_path_factory) -> SmallSearch:
    """The small search, run once with Calculon. Tests take a copy, never this file."""
    path = tmp_path_factory.mktemp("small_search") / "small.duckdb"
    space = SearchSpace(**SMALL_SEARCH)
    events: list[PairProgress] = []
    search_id = run_outer_search(space, db_path=str(path), progress=events.append)
    return SmallSearch(path, space, search_id, events)


@pytest.fixture
def small_search(small_search_template, tmp_path) -> SmallSearch:
    """A copy of the small search for one test."""
    return small_search_template.copy_to(tmp_path)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.duckdb"


@pytest.fixture
def conn(db_path: Path):
    connection = connect(db_path)
    migrate(connection)
    yield connection
    connection.close()
