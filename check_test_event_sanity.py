from events import Events
import colorama

for d in ['GDELT', 'ICEWS14', 'ICEWS18', 'WIKI', 'YAGO']:
    e = Events(d)
    e.update_copy_mask(e.ts_train - 1)
    hist = e.object_copy_mask_dict
    total = 0
    hit = 0
    for ev in e.test_events:
        for s, r, o, _ in ev:
            total += 1
            if o in hist[r][s]:
                hit += 1
    print(colorama.Fore.RED + '{}\thit%: {:4f}\ttotal:{:d}'.format(d, hit / total, total) + colorama.Style.RESET_ALL)
