import typer
from train import train_model, test_model
from data import export_vocab
from data import processSpecialCharsAndKeyWords
app = typer.Typer()

@app.command()
def train(): train_model()

@app.command()
def export_vocabulary(): export_vocab()

@app.command(name="preprocessFile")
def preprocessFile(
    input_file: str = typer.Argument(..., help="Path to the text file you want to filter")
):
    processSpecialCharsAndKeyWords(input_file)

@app.command()
def test(): test_model(10000)

if __name__ == "__main__":
    app()
