import telegram


def send_telegram(photo_path="alert.png"):
    try:
        my_token = "5892768490:AAEmFnyXs-pqtn-8SFQU8dsvmwqZKNA7aFA"
        bot = telegram.Bot(token=my_token)
        bot.sendPhoto(chat_id=5838844371, photo=open(photo_path, "rb"), caption="Có xâm nhập, nguy hiêm!")
        bot.sendMessage
    except Exception as ex:
        print("Can not send message telegram ", ex)

    print("Send sucess")
