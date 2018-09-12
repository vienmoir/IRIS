#! /usr/bin/env python
# -*- coding: utf-8 -*-
#vim:fileencoding=utf-8
import os
from telegram.ext import Updater, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from loadmodel import LoadModel
from cropim import CropIm
from classifyim import Classify

import random

def help(bot, update):
    a = """
    /start - start a chat
    /help - list of existing commands
    /howto - instruction on how to take a photo
    """
    update.message.reply_text(a)

def start(bot, update):
    update.message.reply_text('Send me a picture of a tree inflorescence, please. I will use it to determine a tree species :)')

def get_image(bot, update):
    global model, lb
    file_id = update.message.photo[-1].file_id
    photo = bot.getFile(file_id)
    photo.download(file_id+'.png')
    img = CropIm(file_id+'.png')
    os.remove(file_id+'.png')
    update.message.reply_text(random.choice([
        'Recognition in progress',
        "One moment, I'll check what tree is that",
        'Processing...'
        ]))
    first, second, third = Classify(img, model, lb)
    print("success")
    if first == "none"
        update.message.reply_text("I'm not sure what is it. Please, try another image1")
    else:
        result = label.replace("_", " ")
        result = result.capitalize()
        update.message.reply_text("This is most probably %s. It might also be %s or %s, though!" % (first, second, third))


def reply_text(bot, update):
    update.message.reply_text(random.choice([
         'Send me a picture of inflorescence, please',
         'It\'s better to use this bot in spring!',
         'The weather is great, time to go to the park!',
    ]))

def howto(bot, update):
    h = """
    Go to the tree you wish to recognize (it's better to do in spring!)\nTake a nice shot of the inflorescence. \nMake sure there are no foreign objects on the image.
    Good luck!
    """
    update.message.reply_text(h)

def main():
    token = open("t.txt")
    t = token.read()
    token.close()
    updater = Updater(t)
    updater.dispatcher.add_handler(CommandHandler('start', start))
    updater.dispatcher.add_handler(CommandHandler('help',help))
    updater.dispatcher.add_handler(CommandHandler('howto', howto))
    updater.dispatcher.add_handler(MessageHandler(filters.Filters.photo, get_image))
    updater.dispatcher.add_handler(MessageHandler(filters.Filters.text, reply_text))
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    print("I'm here")
    model, lb = LoadModel()
    print("ok")
    main()