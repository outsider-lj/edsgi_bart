# import xlwt
# from pathlib import Path
# project_dir = Path(__file__).resolve().parent.parent
# print(project_dir)
# workbook = xlwt.Workbook(encoding='utf-8')
# sheet = workbook.add_sheet('Sheet1')
# head = ['num','inputsentences','truesentences','generatedsentences']
# for h in range(len(head)):
#     sheet.write(0, h, head[h])
# test_fileplace=project_dir.joinpath("EmpTransfo/ED/model/test_results.txt")
# test_file=open(test_fileplace,"r",encoding="utf-8")
# sum=1
# for i,line in enumerate(test_file):
#     if i%4==0:
#         sheet.write(sum, 1, line)
#     if i%4==1:
#         sheet.write(sum, 2, line)
#     if i%4==2:
#         sheet.write(sum, 3, line)
#     if i%4==3:
#         sheet.write(sum, 0, line)
#     if i%5==4:
#         sheet.write(sum, 0, sum)
        # sum+=1
# workbook.save(project_dir.joinpath("EmpTransfo/ED/model/test.xls"))

import xlwt
from pathlib import Path
project_dir = Path(__file__).resolve().parent.parent
print(project_dir)
workbook = xlwt.Workbook(encoding='utf-8')
sheet = workbook.add_sheet('Sheet1')
head = ['num','inputsentences','emo_fdbk_true','emo_fdbk_pre','topic_fdbk_true','topic_fdbk_pre','final_sent_true','final_sent_pre']
for h in range(len(head)):
    sheet.write(0, h, head[h])
test_fileplace=project_dir.joinpath("mk-hbart/bart-base/edsgi/test_results.txt")
test_file=open(test_fileplace,"r",encoding="utf-8")
sum=1
for i,line in enumerate(test_file):
    if i%8==0:
        sheet.write(sum, 1, line)
    if i%8==1:
        sheet.write(sum, 2, line)
    if i%8==2:
        sheet.write(sum, 3, line)
    if i%8==3:
        sheet.write(sum, 4, line)
    if i%8==4:
        sheet.write(sum, 5, line)
    if i%8==5:
        sheet.write(sum, 6, line)
    if i % 8 == 6:
        sheet.write(sum, 7, line)
    if i % 8 == 7:
        sheet.write(sum, 0, sum)
    # if i%6==3:
    #     sheet.write(sum, 0, line)
    # if i%5==4:
    #     sheet.write(sum, 0, sum)
        sum+=1
workbook.save(project_dir.joinpath("mk-hbart/bart-base/edsgi/test_top5.xls"))