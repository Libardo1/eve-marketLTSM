import requests
parameters = {"char_name":"demo","region_ids":10000002,"type_ids":238,"days":50}
r = requests.get("http://eve-marketdata.com/api/item_history2.json?",params=parameters)
print(r.url)
print(r)
print(r.text)

