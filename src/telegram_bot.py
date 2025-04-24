from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
import torch
from classical_model_bert import tokenizer, CyberbullyingDetector

max_len = 31

NUM_CLASSES = 2 
HIDDEN_DIM = 128



cl_model = CyberbullyingDetector(HIDDEN_DIM, NUM_CLASSES)

cl_model.load_state_dict(torch.load('data\cl_bert_weights.pth', weights_only=True, map_location=torch.device('cpu')))
cl_model.eval()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Hello! I’m a bot detecting cyberbullying message. Send me message and I'll check it")

async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text_received = update.message.text
    input_tensor = tokenizer(
        text_received,
        max_length=31,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = input_tensor['input_ids']
    attention_mask = input_tensor['attention_mask']
       
    preds = cl_model(input_ids=input_ids, attention_mask=attention_mask)
    y_pred_test = torch.argmax(preds, dim=1)
    output = "Not cyberbullying" if y_pred_test==0 else "Cyberbullying"
    await update.message.reply_text(output)

def main():
    token = "my token"
    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo))

    app.run_polling()

if __name__ == "__main__":
    main()