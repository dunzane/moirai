import typer
from .cmds import train,val

app = typer.Typer()
app.command(name="train")(train.run)
app.command(name="val")(val.run)

if __name__ == "__main__":
    app()


