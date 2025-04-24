from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import torch
from model import ClassicalModel
from preprocessing_for_bot import clean_tweet, Tokenize

MAX_LEN = 31
NUM_CLASSES = 2 
HIDDEN_DIM = 128
LSTM_LAYERS = 2
VOCAB_SIZE = 10000
EMBEDDING_DIM = 100

cl_model = ClassicalModel(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES, LSTM_LAYERS)
cl_model.load_state_dict(torch.load('data/cl_lstm_weights.pth', weights_only=True, map_location=torch.device('cpu')))
cl_model.eval()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I’m a bot detecting cyberbullying messages. Send me a message, and I'll check it.")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_received = update.message.text
    cleaned_text = clean_tweet(text_received)

    _, message_indices = Tokenize([cleaned_text], 16)

    hidden = cl_model.init_hidden(batch_size=1)

    with torch.no_grad():
        output, _ = cl_model(message_indices, hidden)
        predicted_class = output.argmax(dim=1).item()

    if predicted_class == 0:
        output = "Not cyberbullying"
    else:
        output = "Cyberbullying"

    await update.message.reply_text(output)

def main():
    token = "my token" 
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    app.run_polling()

if __name__ == "__main__":
    main()