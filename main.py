import typer
from train import train_model
from data import export_vocab

app = typer.Typer()

@app.command()
def train(): train_model()

@app.command()
def export_vocabulary(): export_vocab()

if __name__ == "__main__":
    app()
