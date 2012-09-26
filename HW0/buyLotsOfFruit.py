def buyLotsOfFruit(orderList):
	fruitPrices = {'apples':2.00, 'oranges': 1.50, 'pears': 1.75,
              'limes':0.75, 'strawberries':1.00}
	total = 0
	for fruit, pound in orderList:
		if fruit not in fruitPrices:
			print "I'm sorry, you ordered something that isn't for sale"
			return None
		else:
			total += pound*fruitPrices[fruit]
	print "The total cost is %f" % (total)

if __name__ == '__main__':
	buyLotsOfFruit([('apples', 2.0), ('pears', 3.0), ('limes', 4.0)])