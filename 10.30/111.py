# coding:utf-8

import xlrd
import xlsxwriter

xlsxbook = xlrd.open_workbook('section5.xlsx')
xlsxsheet = xlsxbook.sheet_by_index(0)

mobile = xlsxsheet.col_values(0)
x1 = xlsxsheet.col_values(1)
x2 = xlsxsheet.col_values(2)
x3 = xlsxsheet.col_values(3)
x4 = xlsxsheet.col_values(4)
x5 = xlsxsheet.col_values(5)
x6 = xlsxsheet.col_values(6)
x7 = xlsxsheet.col_values(7)
x8 = xlsxsheet.col_values(8)
x9 = xlsxsheet.col_values(9)
x10 = xlsxsheet.col_values(10)
row_num = xlsxsheet.nrows

workbook = xlsxwriter.Workbook('selection.xlsx')
worksheet = workbook.add_worksheet('selection')

worksheet.write(0, 0, mobile[0])
worksheet.write(0, 1, x1[0])
worksheet.write(0, 2, x2[0])
worksheet.write(0, 3, x3[0])
worksheet.write(0, 4, x4[0])
worksheet.write(0, 5, x5[0])
worksheet.write(0, 6, x6[0])
worksheet.write(0, 7, x7[0])
worksheet.write(0, 8, x8[0])
worksheet.write(0, 9, x9[0])
worksheet.write(0, 10, x10[0])

formal = '123456789'

count = 0
for row in range(row_num):
    print row
    for e in x8[row]:
        if e in formal:
            count += 1
            worksheet.write(count, 0, mobile[row])
            worksheet.write(count, 1, x1[row])
            worksheet.write(count, 2, x2[row])
            worksheet.write(count, 3, x3[row])
            worksheet.write(count, 4, x4[row])
            worksheet.write(count, 5, x5[row])
            worksheet.write(count, 6, x6[row])
            worksheet.write(count, 7, x7[row])
            worksheet.write(count, 8, x8[row])
            worksheet.write(count, 9, x9[row])
            worksheet.write(count, 10, x10[row])
            break
        else:
            pass

workbook.close()