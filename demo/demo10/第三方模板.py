import schedule
import time
def job():
    print('卓伟攀，要开始工作了')
schedule.every(3).seconds.do(job)
while True:
    schedule.run_pending()
    time.sleep(1)

