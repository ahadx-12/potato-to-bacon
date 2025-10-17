import click
from potatobacon.sdk.client import PBClient, PBConfig


@click.group()
@click.option("--base-url", default="http://localhost:8000", help="API base URL")
@click.pass_context
def ptb(ctx, base_url):
    ctx.obj = PBClient(PBConfig(base_url=base_url))


@ptb.command()
@click.argument("dsl_file", type=click.Path(exists=True))
@click.pass_obj
def translate(client, dsl_file):
    txt = open(dsl_file).read()
    res = client.translate(txt)
    click.echo(res)


@ptb.command()
@click.argument("dsl_file", type=click.Path(exists=True))
@click.option("--domain", default="classical")
@click.pass_obj
def validate(client, dsl_file, domain):
    txt = open(dsl_file).read()
    res = client.validate(txt, domain=domain)
    click.echo(res)


@ptb.command()
@click.argument("dsl_file", type=click.Path(exists=True))
@click.option("--name", default="compute")
@click.pass_obj
def codegen(client, dsl_file, name):
    txt = open(dsl_file).read()
    code = client.codegen(txt, name=name)
    click.echo(code)


@ptb.command()
@click.argument("dsl_file", type=click.Path(exists=True))
@click.option("--domain", default="classical")
@click.pass_obj
def manifest(client, dsl_file, domain):
    txt = open(dsl_file).read()
    res = client.manifest(txt, domain=domain)
    click.echo(res)


if __name__ == "__main__":
    ptb()
