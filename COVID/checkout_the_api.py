#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests, json
from pprint import pprint
url 		= 'http://cslab241.cs.aueb.gr:5000/just_the_json'
data 		= {
	'question'  : 'What do we know about Covid-19 vaccines ?',
	'section'   : ''
}
r = requests.post(url, json=data)
print(r.status_code)
pprint(r.json())
