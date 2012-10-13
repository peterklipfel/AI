fruitPrices = {'apples': 2.00, 'oranges': 1.50, 'pears': 1.75}

def buyFruit(fruit, numPounds):
	if fruit not in fruitPrices:
		print "Sorry, we don't have %s in stock" % (fruit)
	else:
		cost = fruitPrices[fruit]*numPounds
		print "That costs %f dollars" % (cost)

# Main Function

if __name__ == '__main__':
	buyFruit('apples', 23)
	buyFruit('coconuts', 83)