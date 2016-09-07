#Hadoop 入门系列
应该说，hadoop内容蛮多的，譬如hive,hdfs,mapreduce,维基百科上这样介绍的，开源框架，分布式存储，分布式处理，其核心有两部分，一个是HDFS，一个是MapReduce。大致处理流程是，Hadoop将文件分成很多blocks，然后将这些块分布到一个集群中的各个节点上去，然后将打包好的代码分发到各个节点然后处理待处理的数据。这与并行计算有些不同，充分的利用了数据的局部性。
现在安装Hadoop的时候，必然要编译一遍库文件才可以使用了，而且我们一般安装的是其框架，其实我们可以往这个框架里面方很多东西，都应该形成了一个完整的生态系统了，譬如Spark, pig, hive, Flume, HBASE, storm等等。而这个框架本身有四个部分组成。
>Hadoop Common:这里包含了一些Hadoop模块的工具和库文件。
>HDFS:这个应该是分布式文件系统的一些东西，譬如节点之间的传输，文件如何切分。
>YARN:资源管理和资源调度的平台
>MapReduce:对于大型数据处理的MapReduce的程序模块

先说说mapreduce，mapreduce是hadoop的计算的程序模块，从编程者的角度，这东西就包括了两个模块，Map方法，Reduce方法。Map方法主要是用来作为数据的输入，通常数据被分成很多key，对于每个key我们都有相应的value值。Reduce方法是执行总结操作，对每一个key值作为一组进行处理，但是并不是之所有相同的key值，是同一个节点的相同key值。中间还有个shuffle，对用户是隐藏的，用来重新分布map()函数的输出的。总的说来，使用MR有五个步骤：
>1.准备数据作为Map()方法的输入，输入一般是<key1,value1>形式，key1常常是我们的数据，对于wordCount来说，key1就是单词，value1全部赋值为1。
>2.继承父类的的Map()函数，并且实现实现相关功能，输入是1步骤的<key1,value1>，输出是<key2,value2>。
>3.shuffle操作是将Map的输出<key2,value2>重新分配给Reducer，不太清楚具体怎么实现的。
>4.继承父类的Reduce()，对每一个key2都执行一次Reduce()操作，也就是统计。
>5.最后是集合所有的Reduce输出，并且按照key2排序。

有人总结了一个逻辑流程如下：
>raw data -> <key1, value1> ->Map(key1, value1)->list(key2,value2)


















