"""Normalize only the optional W&B blocks for historical notebook guards."""
from copy import deepcopy


def without_wandb(cell):
    cell = deepcopy(cell)
    source = ''.join(cell['source'])
    for label, indent, replacement in (('settings', '', ''), ('reporter', '    ', '    run(*args)\n')):
        start = indent + '# BEGIN baseline W&B ' + label + '\n'
        end = indent + '# END baseline W&B ' + label + '\n'
        if start in source:
            before, block = source.split(start, 1)
            _, after = block.split(end, 1)
            source = before + replacement + after
    cell['source'] = source.splitlines(keepends=True)
    return cell
