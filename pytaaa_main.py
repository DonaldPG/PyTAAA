import click

from run_pytaaa import run_pytaaa

@click.command()
@click.option(
    '--json', 'json_fn',
    type=click.Path(exists=True),
    help='Path to the JSON file with PyTAAA parameters'
)
def main(json_fn):
    if json_fn:
        run_pytaaa(json_fn)
    else:
        click.echo(
            'Please specify a JSON file with PyTAAA parameters'
            ' using the --json tag.'
        )


if __name__ == '__main__':
    main()
