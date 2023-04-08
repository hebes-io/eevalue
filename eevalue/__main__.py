# -*- coding: utf-8 -*-
# Copyright (c) Hebes Intelligence Private Company

# This source code is licensed under the Apache License, Version 2.0 found in the
# LICENSE file in the root directory of this source tree.

import click

from eevalue.cli.backtest import backtest
from eevalue.cli.calibrate import calibrate
from eevalue.cli.compare import compare
from eevalue.cli.ordc import ordc
from eevalue.cli.preprocess import preprocess
from eevalue.cli.replay import replay
from eevalue.cli.runs import runs
from eevalue.cli.simulate import simulate

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.group(context_settings=CONTEXT_SETTINGS, name="eevalue")
def cli():
    """Command line tool for running the eevalue package."""


cli.add_command(backtest)
cli.add_command(calibrate)
cli.add_command(compare)
cli.add_command(ordc)
cli.add_command(preprocess)
cli.add_command(replay)
cli.add_command(runs)
cli.add_command(simulate)


def main():
    cli()
