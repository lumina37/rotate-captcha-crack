from collections.abc import Callable, Iterator, Sequence
from typing import Generic, Protocol, SupportsIndex, TypeVar, overload, runtime_checkable

TArg = TypeVar('TArg', contravariant=True)  # Arg Type
TRet = TypeVar('TRet', covariant=True)  # Return Type


@runtime_checkable
class HasFns(Protocol):
    fns: list[Callable]


class FnWrap(Generic[TArg, TRet]):
    """
    Wrapper of pure function which allows you to pre-compose several funcions using pipeline operator.

    Args:
        fn (Callable[[TArg], TRet]): a pure function

    Example:
        `precomposed_funcs = FnWrap(func00) | func01 | func02` \\
        `another_funcs = FnWrap(func10) | precomposed_funcs | func12`
    """

    def __init__(self, fn: Callable[[TArg], TRet]) -> None:
        self.fns = [fn]

    def __call__(self, item: TArg) -> TRet:
        for fn in self.fns:
            item = fn(item)
        return item

    def __or__[NewRT](self, rhs: Callable[[TRet], NewRT]) -> "FnWrap[TArg, NewRT]":
        if isinstance(rhs, HasFns):
            self.fns += rhs.fns
        else:
            self.fns.append(rhs)
        return self


TypeFnWrap = TypeVar('TypeFnWrap', bound=FnWrap)


class IteratorRoot(Iterator[TRet]):
    def __init__(self, iterator: Iterator[TRet]) -> None:
        self.iterator = iterator
        self.fns = []

    def __or__[NewTRet](self, fn: Callable[[TRet], NewTRet]) -> "IteratorRoot[NewTRet]":
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
    def __init__(self, sequence: Sequence[TRet]) -> None:
        self.sequence = sequence
        self.fns = []

    def __or__[NewTRet](self, fn: Callable[[TRet], NewTRet]) -> "SequenceRoot[NewTRet]":
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
    def __getitem__(self, idx: slice) -> "SequenceRoot[TRet]": ...

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
