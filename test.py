
import mysql.connector

cnx = mysql.connector.connect(
    host="xaifwdb.c0gj2xdlz1ck.us-east-2.rds.amazonaws.com",
    user="root",
    password="xaidb.2022",
    database="xaifw"
)

cursor = cnx.cursor()

query = ("SELECT * FROM explanation")

cursor.execute(query)
# print selected data
for (model_name, method_name, data_set_name, data_set_group_name, task_name, explanation) in cursor:
    print(model_name, method_name, data_set_name, data_set_group_name, task_name, explanation)

cursor.close()
cnx.close()
