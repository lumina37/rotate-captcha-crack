from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Generic, Protocol, SupportsIndex, TypeVar, overload, runtime_checkable

TArg = TypeVar('TArg', contravariant=True)  # Arg Type
TRet = TypeVar('TRet', covariant=True)  # Return Type
NewTRet = TypeVar('NewTRet', covariant=True)  # New Return Type


@runtime_checkable
class HasFns(Protocol):
    fns: list[Callable]


class FnWrap(Generic[TArg, TRet]):
    """
    Wrapper of pure function which allows you to pre-compose several funcions using pipeline operator.

    Args:
        fn (Callable[[TArg], TRet]): a pure function

    Example:
        ```
        precomposed_funcs = FnWrap(func00) | func01 | func02
        another_funcs = FnWrap(func10) | precomposed_funcs | func12
        ```
    """

    def __init__(self, fn: Callable[[TArg], TRet]) -> None:
        self.fns = [fn]

    def __call__(self, item: TArg) -> TRet:
        for fn in self.fns:
            item = fn(item)
        return item

    def __or__(self, rhs: Callable[[TRet], NewTRet]) -> FnWrap[TArg, NewTRet]:
        if isinstance(rhs, HasFns):
            self.fns += rhs.fns
        else:
            self.fns.append(rhs)
        return self


TypeFnWrap = TypeVar('TypeFnWrap', bound=FnWrap)


class IteratorRoot(Generic[TRet], Iterator[TRet]):
    """
    Iterator supporting pipeline operator `|`.

    Args:
        iterator (Iterator[TRet]): base iterator

    Example:
        ```
        path_filter = lambda p: p if p.startswith('cat') else None
        path_iterator = IteratorRoot(path_iterator)
        new_path_iterator = root_iterator | path_filter
        for cat_path in new_iterator:
            # do something
        ```
    """

    def __init__(self, iterator: Iterator[TRet]) -> None:
        self.iterator = iterator
        self.fns = []

    def __or__(self, fn: Callable[[TRet], NewTRet]) -> IteratorRoot[NewTRet]:
        if isinstance(fn, HasFns):
            self.fns += fn.fns
        else:
            self.fns.append(fn)
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


class SequenceRoot(Sequence[TRet]):
    """
    Sequence supporting pipeline operator `|`.
    The functions will only be applied after you call `__getitem__`, which is also called *lazy execution*.

    Args:
        sequence (Sequence[TRet]): base sequence

    Example:
        ```
        from torchvision.transforms.functional import rgb_to_grayscale

        images = SequenceRoot(images)
        gray_images = images | rgb_to_grayscale
        for image in gray_images:
            # do something
        ```
    """

    def __init__(self, sequence: Sequence[TRet]) -> None:
        self.sequence = sequence
        self.fns = []

    def __or__(self, fn: Callable[[TRet], NewTRet]) -> SequenceRoot[NewTRet]:
        if isinstance(fn, HasFns):
            self.fns += fn.fns
        else:
            self.fns.append(fn)
        return self

    def __len__(self) -> int:
        return len(self.sequence)

    @overload
    def __getitem__(self, idx: SupportsIndex) -> TRet: ...

    @overload
    def __getitem__(self, idx: slice) -> SequenceRoot[TRet]: ...

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            newroot = SequenceRoot(self.sequence[idx])
            newroot.fns = self.fns.copy()
            return newroot

        else:
            item = self.sequence.__getitem__(idx)
            for fn in self.fns:
                item = fn(item)
            return item
