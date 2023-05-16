#!/usr/bin/env python
# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

# Standard
from pathlib import Path
import os
import shutil

# Third Party
import click

HOME = Path.home()
path = f"{HOME}/.cache/huggingface"


@click.group()
def cli():
    pass


@click.command()
@click.option("--confirm", is_flag=True, help="Confirm you want to delete the cache.")
def clean(confirm):
    """Clean the huggingface cache"""
    if confirm:
        try:
            shutil.rmtree(path)
            print("---> The huggingface cache is removed")
        except OSError as x:
            print(f"Error occured: {path} : {x.strerror}")
    else:
        print("You need to --confirm to delete the cache.")


@click.command()
def start():
    """Start the server"""
    if os.system("/usr/bin/env python app.py"):
        raise RuntimeError("Failed to start the gradio server")


@click.command()
def setup():
    """Install the needed dependancies"""
    print("TODO: This will set up the dependacies")


@click.command()
def add():
    """Add models to the application"""
    print("TODO: This will add models to the application")


cli.add_command(clean)
cli.add_command(start)
cli.add_command(setup)

if __name__ == "__main__":
    cli()
