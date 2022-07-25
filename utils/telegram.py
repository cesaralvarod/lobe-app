import requests


TELEGRAM_TOKEN_BOT = "5504264647:AAH8hVpQSdYmDO3CTbNXkZfyvTwtBFp1LQ4"
TELEGRAM_CHAT_ID = "-779844543"
TELEGRAM_API = "https://api.telegram.org/bot"


def send_image_telegram(filename, caption):
    url_request = TELEGRAM_API + TELEGRAM_TOKEN_BOT + "/sendPhoto"
    data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
    files = {"photo": (filename, open(filename, "rb"))}

    try:
        res = requests.post(url_request, data=data, files=files).json()
        if(res['ok']):
            print('Message sent')
        else:
            print(res)
    except Exception as exc:
        print(exc)
