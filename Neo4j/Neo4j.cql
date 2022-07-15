##############################
# Scrips to generate graph in Neo4j
##############################

# Add nodes

# sets 
LOAD CSV WITH HEADERS from 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRd7Rh1eD02HClVfOouM1iH23sFzMZDwJ31mmeXcvatjN0EjhNBJSwc0VUkHRqYScTRVqblbbxXxjDO/pub?output=csv' as row with row where row.source is not null
MERGE (s:Sets {sets: row.source})
MERGE (p:Product {prod: row.target})
MERGE (s)-[:Include]->(p)

# theme
LOAD CSV WITH HEADERS from 'https://docs.google.com/spreadsheets/d/e/2PACX-1vT9yv7GAbYTbwOAXavf7BglCA1t5Mxl9AYShREjw_cmKow7pJV6Yuz3YHTyDazMVBei17p9XJsp003h/pub?output=csv' as row with row where row.source is not null
MERGE (t:Theme {theme: row.source})
MERGE (c:Series {series: row.target})
MERGE (t)-[:Include]->(c)

# Item# 
LOAD CSV WITH HEADERS from 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRNReHGwl999X4XHLFI95aoQmjPeDzqVgU6Oo-_3cv6YgxVAMU6R3k19r2-8R43w6agxyTUk5Dyq2aG/pub?output=csv' as row with row where row.source is not null
MERGE (i:Item {item: row.source})
MERGE (p:ProductCode {code: row.target})
MERGE (i)-[:Include]->(p)

# Age 
LOAD CSV WITH HEADERS from 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRs2BRQYrnsCeBRwWYItPENNOItgyniFnzuovhMyj533CcX6hSu74z_1VGPo7AX8-vNJspCzN37_JIr/pub?output=csv' as row with row where row.source is not null
MERGE (a:Age {age: row.source})
MERGE (c:AgeRange {ageRan: row.target})
MERGE (a)-[:Include]->(c)

# Count 
LOAD CSV WITH HEADERS from 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQC4WF2mhrOutjMkL67nywtUvru_EhGAJ0_iBB9pp-CJJFaaumegN4uiPdUFOkX_t5vM8jLNDFtgwrn/pub?output=csv' as row with row where row.source is not null
MERGE (c:Count {count: row.source})
MERGE (p:PieceCount {piece: row.target})
MERGE (c)-[:Include]->(p)

# Price 
LOAD CSV WITH HEADERS from 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRHIfnMZQSC0ZlURxV3Yrw_epOHd_EC0wpXNkmmKafT2Vh0K2SHqdgi5vJdds8jnQkzZzWT3A0ZrbFx/pub?output=csv' as row with row where row.source is not null
MERGE (t:Pricetag {pricetag: row.source})
MERGE (c:Price {price: row.target})
MERGE (t)-[:Include]->(c)

# Discount 
LOAD CSV WITH HEADERS from 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRszdElZp7IPtDMWexC1IuwxI4-NyLTpG3CY_F7JuwaViTFzt3X-TWedyXmlsRkam6rwbKRyYpJb2DN/pub?output=csv' as row with row where row.source is not null
MERGE (d:Discount {discount: row.source})
MERGE (c:Sale {sale: row.target})
MERGE (d)-[:Include]->(c)

# InStock 
LOAD CSV WITH HEADERS from 'https://docs.google.com/spreadsheets/d/e/2PACX-1vR9hwEzOoCPdWwo-vK6_O1QxKYx1vgGtlZ2W0lZDTMUlNMNycYi-7Tsv_iSuGzs2HQz0EaS1YQfhE9J/pub?output=csv' as row with row where row.source is not null
MERGE (t:InStock {instock: row.source})
MERGE (c:Availability {avail: row.target})
MERGE (t)-[:Include]->(c)


LOAD CSV WITH HEADERS from ‘https://docs.google.com/spreadsheets/d/e/2PACX-1vQyXAhZS7KvYHAENGp2TZVXKoEf96BWZOoewSVHK2Q8_tdtQS2KxQX69y95HzXthzRuAdXRmnu6P1Iy/pub?output=csv’ as row with row where row.source is not null
MERGE (p:Product {prod: row.source})
MERGE (c:Series {series: row.target})
MERGE (p)-[:belong]->(c)
 
LOAD CSV WITH HEADERS from ‘https://docs.google.com/spreadsheets/d/e/2PACX-1vQCpTW_pzIUO0gCYZ-ox09IFhQGva_9Pn7BLBfEqzu7KC75Zoqwqc6E8nzTGIS23FZWfjg7iISNnazf/pub?output=csv’ as row with row where row.source is not null
MERGE (p:ProductCode {code: row.source})
MERGE (c:PieceCount {piece: row.target})
MERGE (p)-[:piececount]->(c)
 
LOAD CSV WITH HEADERS from ‘https://docs.google.com/spreadsheets/d/e/2PACX-1vTYRMRuUeWWHO978Udyj9qcaUdE54RYBr6JDZfvelU5Jak-6QMbLhfT34OoRWsGyybFvaFiHa8rPmoP/pub?output=csv' as row with row where row.source is not null
MERGE (p:ProductCode {code: row.source})
MERGE (c:Price {price: row.target})
MERGE (p)-[:current_price]->(c)
 
LOAD CSV WITH HEADERS from ‘https://docs.google.com/spreadsheets/d/e/2PACX-1vT2WDnQeJOraDVrkY8J4_De9J6j6Pr76NZ4u23lm6tgRXlMOhwV_GZqa9K5ZnilPi1br3yKQ1cq-WNk/pub?output=csv’ as row with row where row.source is not null
MERGE (p:Product {prod: row.source})
MERGE (c:ProductCode {code: row.target})
MERGE (p)-[:product_code]->(c)
 
LOAD CSV WITH HEADERS from ‘https://docs.google.com/spreadsheets/d/e/2PACX-1vRVEryCwD4MVk04moC3QOKp4n7Ciy_UWjwOusJWQX8rtXqNIhM8z7dHAy5WKugE12V94p-77OvrvQCk/pub?output=csv' as row with row where row.source is not null
MERGE (p:ProductCode {code: row.source})
MERGE (c:AgeRange {ageRan: row.target})
MERGE (p)-[:AgeRange]->(c)

LOAD CSV WITH HEADERS from ‘https://docs.google.com/spreadsheets/d/e/2PACX-1vRSogYmscvkAHMEd7kGCxAMBrnTyuWeUoum3yp2fWAtBwOyIp7rlb4Gbl33a9qfX96aLlnd_bMit7kU/pub?output=csv’ as row with row where row.source is not null
MERGE (p:ProductCode {code: row.source})
MERGE (c:Availability {avail: row.target})
MERGE (p)-[:availability]->(c)
 
LOAD CSV WITH HEADERS from ‘https://docs.google.com/spreadsheets/d/e/2PACX-1vRrsfoUalrqRJSvIm97zzeAqi4onEjvNg0M_nfHwOUClVk4r2eOZLS6tNBCLRo2bbd9NFYHwgUbj3GA/pub?output=csv' as row with row where row.source is not null
MERGE (p:ProductCode {code: row.source})
MERGE (c:Sale {sale: row.target})
MERGE (p)-[:sale]->(c)

# View all 
MATCH (n) RETURN n

# Delete a ll nodes 
MATCH (n) DETACH DELETE n