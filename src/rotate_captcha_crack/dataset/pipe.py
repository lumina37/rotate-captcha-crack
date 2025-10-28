from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import (
    Generic,
    Protocol,
    SupportsIndex,
    TypeVar,
    overload,
    runtime_checkable,
)

TArg = TypeVar("TArg", contravariant=True)  # Arg Type
TRet = TypeVar("TRet", covariant=True)  # Return Type
NewTRet = TypeVar("NewTRet", covariant=True)  # New Return Type


@runtime_checkable
class HasFns(Protocol):
    fns: list[Callable]


class FnSupportsPipe(Generic[TArg, TRet]):
    """
    Wrapper of pure function which allows you to pre-compose several funcions using pipe operator.

    Example:
        ```
        precomposed_funcs = func00 | FnSupportsPipe() | func01 | func02
        another_funcs = func10 | precomposed_funcs | func12
        ```
    """

    def __init__(self) -> None:
        self.fns = []

    def __call__(self, item: TArg) -> TRet:
        for fn in self.fns:
            item = fn(item)
        return item

    def __or__(self, rhs: Callable[[TRet], NewTRet]) -> FnSupportsPipe[TArg, NewTRet]:
        if isinstance(rhs, HasFns):
            self.fns += rhs.fns
        else:
            self.fns.append(rhs)
        return self

    def __ror__(self, fn: Callable[[TArg], TRet]) -> FnSupportsPipe[TArg, TRet]:
        self.fns.append(fn)
        return self


TypeFnSupportsPipe = TypeVar("TypeFnSupportsPipe", bound=FnSupportsPipe)


class IterSupportsPipe(Generic[TRet], Iterator[TRet]):
    """
    Make the lhs iterator support pipe operator `|`.

    Example:
        ```
        path_filter = lambda p: p if p.startswith('cat') else None
        new_path_iterator = path_iterator | IterSupportsPipe() | path_filter
        for cat_path in new_iterator:
            # do something
        ```
    """

    def __init__(self) -> None:
        self.iterator = iter([])
        self.fns = []

    def __or__(self, fn: Callable[[TRet], NewTRet]) -> IterSupportsPipe[NewTRet]:
        if isinstance(fn, HasFns):
            self.fns += fn.fns
        else:
            self.fns.append(fn)
        return self

    def __ror__(self, iterator: Iterator[TRet]) -> IterSupportsPipe[TRet]:
        self.iterator = iterator
        return self

    def __iter__(self) -> Iterator[TRet]:
        return self

    def __next__(self) -> TRet:
        while 1:
            item = next(self.iterator)

            skip = False
            for fn in self.fns:
                item = fn(item)
                if item is None:
                    skip = True
                    break

            if skip:
                continue

            return item


class SeqSupportsPipe(Sequence[TRet]):
    """
    Make the lhs sequence support pipe operator `|`.
    The functions will only be applied after you call `__getitem__`, which is also called *lazy execution*.

    Example:
        ```
        from torchvision.transforms.functional import rgb_to_grayscale

        images = [...]
        gray_images = images | SeqSupportsPipe() | rgb_to_grayscale
        for gray_image in gray_images:
            # do something
        ```
    """

    def __init__(self) -> None:
        self.sequence = []
        self.fns = []

    def __or__(self, fn: Callable[[TRet], NewTRet]) -> SeqSupportsPipe[NewTRet]:
        if isinstance(fn, HasFns):
            self.fns += fn.fns
        else:
            self.fns.append(fn)
        return self

    def __ror__(self, sequence: Sequence[TRet]) -> SeqSupportsPipe[TRet]:
        self.sequence = sequence
        return self

    def __len__(self) -> int:
        return len(self.sequence)

    @overload
    def __getitem__(self, idx: SupportsIndex) -> TRet: ...

    @overload
    def __getitem__(self, idx: slice) -> SeqSupportsPipe[TRet]: ...

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            newroot = SeqSupportsPipe(self.sequence[idx])
            newroot.fns = self.fns.copy()
            return newroot

        else:
            item = self.sequence.__getitem__(idx)
            for fn in self.fns:
                item = fn(item)
            return item
