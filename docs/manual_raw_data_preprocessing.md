# Changes to `london_wards_new.json` file

The following feature names where changed:

- The `cmwd11nm` field was changed from "Roehampton and Putney Heath" to "Roehampton"
`{ "objectid": 7650,
"cmwd11cd": "E36007650",
"lad11cd": "E09000032",
"lad11nm": "Wandsworth" } `

- The `cmwd11nm` field was changed from "St Katharine's and Wapping" to "St Katherine's and Wapping"
`{"cmwd11cd": "E36007616",
  "cmwd11nm": "St Katharine's and Wapping",
  "cmwd11nmw": null,
  "lad11cd": "E09000030",
  "lad11nm": "Tower Hamlets"} `

# Changes to `ethnic-group-ward-2001.csv` file

The following row was changed:
`borough: City of London\
 ward: City of London\
 pop2001: 7177'

 The ward "City of London" was removed and instead wards "Aldersgate", "Aldgate", "Bassishaw" and "Bread Street" (all part of the City of London) were added
 The population (7177) of the deleted ward was equally distributed to the four new wards (1,794.25 ~= 1,794 each)
