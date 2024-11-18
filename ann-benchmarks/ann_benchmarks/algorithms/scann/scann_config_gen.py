#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 版权所有 (c) 华为技术有限公司 2012-2020

import sys
import copy
import logging

pre = '''float:
  angular:
  - base_args: {}
    constructor: Scann
    disabled: false
    docker_tag: ann-benchmarks-scann
    module: ann_benchmarks.algorithms.scann
    name: scann
    run_groups:\n'''
end = '''  euclidean:
  - base_args: {}
    constructor: Scann
    disabled: false
    docker_tag: ann-benchmarks-scann
    module: ann_benchmarks.algorithms.scann
    name: scann
    run_groups:
      scann1:
        args: [[600], [.nan], [2], [squared_l2]]
        query_args: [[[4, 40], [3, 30], [6, 60], [8, 74], [9, 78], [10, 82], [11,
              85], [13, 100], [16, 120], [20, 140], [30, 180], [35, 240], [50, 360]]]
      scann2:
        args: [[2000], [.nan], [4], [squared_l2]]
        query_args: [[[10, 100], [15, 140], [25, 160], [35, 190], [40, 200], [45,
              220], [50, 240], [60, 250], [70, 300], [80, 400], [100, 500], [120,
              600], [150, 800], [200, 900]]]
      scann3:
        args: [[100], [.nan], [4], [squared_l2]]
        query_args: [[[2, 20], [3, 20], [3, 30], [4, 30], [5, 40], [8, 80]]]\n'''

multi = [[1, 30], [2, 30], [4, 30], [8, 30], [30, 120], [35, 100], [40, 80],
        [45, 80], [50, 80], [55, 95], [60, 110], [65, 110], [75, 110],
        [90, 110], [110, 120], [130, 150], [150, 200], [170, 200], [200, 300],
        [220, 500], [250, 500], [310, 300], [400, 300], [500, 500], [800, 1000]]

tail = [[800, 1000]]


def main(argv):
    if len(argv) <= 1:
        logging.warning("Please specify the number of tasks to be configured")
        return
    tasks = int(argv[1])

    args = multi
    if len(argv) > 2 and argv[2] == "tail":
        args = tail

    data = [pre]
    for i in range(11):
        for j in range(30):
            num = i * 30 + j + 1
            if num <= tasks:
                data.append("      scann{}:\n".format(num))
                data.append("        args: [[{}], [0.2], [2], [dot_product]]\n"
                      .format(2000 + i))
                arg = copy.deepcopy(args)
                for elm in arg:
                    elm[0] = elm[0] + j
                    elm[1] = elm[1] - j

                data.append("        query_args: [{}]\n".format(arg))
    data.append(end)
    # print(''.join(data))
    if len(argv) > 2 and argv[2] == "tail":
        output = "config-{}-tail.yml".format(tasks)
    else:
        output = "config-{}.yml".format(tasks)
    with open(output, 'w') as f:
        f.write(''.join(data))


if __name__ == '__main__':
    main(sys.argv)
