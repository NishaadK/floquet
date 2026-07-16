from collections.abc import Iterable

import pathos


def _collect(iterator, total, progress, desc, progress_bar, advance_per_item):
    """Materialize an iterator, optionally driving a progress bar.

    If ``progress_bar`` (a tqdm instance) is supplied, it is advanced by
    ``advance_per_item`` for each completed item and is *not* closed here (the
    caller owns it, so it can persist across many calls). Otherwise, if
    ``progress`` is True, a standalone tqdm bar (or textual counter) is shown.
    Results are returned in input order regardless.
    """
    if progress_bar is not None:
        results = []
        for item in iterator:
            results.append(item)
            progress_bar.update(advance_per_item)
        return results
    if not progress:
        return list(iterator)
    try:
        from tqdm.auto import tqdm

        return list(tqdm(iterator, total=total, desc=desc, leave=False))
    except ImportError:
        results = []
        label = desc or "progress"
        for i, item in enumerate(iterator, start=1):
            results.append(item)
            print(f"\r{label}: {i}/{total}", end="", flush=True)
        print()
        return results


def parallel_map(
    num_cpus: int,
    func: callable,
    parameters: Iterable,
    progress: bool = False,
    desc: str | None = None,
    progress_bar=None,
    advance_per_item: int = 1,
) -> list:
    """Map ``func`` over ``parameters``, optionally in parallel and with a bar.

    Parameters:
        num_cpus: number of worker processes (1 => serial, no pool overhead).
        func: function to apply to each element of ``parameters``.
        parameters: iterable of inputs.
        progress: if True (and no ``progress_bar`` is given), display a
            standalone tqdm bar (falls back to a textual counter without tqdm).
        desc: label for the standalone progress bar.
        progress_bar: an existing tqdm bar to advance instead of creating one;
            lets a caller share a single persistent bar across many calls.
        advance_per_item: amount to advance ``progress_bar`` per completed item.

    Results are always returned in the original input order.
    """
    params = list(parameters)
    total = len(params)
    if num_cpus == 1:
        return _collect(
            map(func, params), total, progress, desc, progress_bar, advance_per_item
        )

    with pathos.pools.ProcessPool(nodes=num_cpus) as pool:
        # imap yields results in input order as workers finish, so the bar
        # advances live; must be consumed inside the `with` block.
        return _collect(
            pool.imap(func, params),
            total,
            progress,
            desc,
            progress_bar,
            advance_per_item,
        )
