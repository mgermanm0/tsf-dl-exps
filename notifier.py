import requests

def telegram_bot_sendtext(bot_message):

   bot_token = None
   bot_chatID = None
   if bot_token is None or bot_chatID is None:
      return "No hay token en notifier.py"
   send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

   response = requests.get(send_text)

   return response.json()