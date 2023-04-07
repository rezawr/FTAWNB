class FTAWNB:
    def __init__(self, type):
        if type == 'text':
            self.type = type
        else:
            raise Exception("type is wrong")

    def buildClassifier(self, attribute, category):
        self.attribute = attribute
        self.category = category

        bagOfWord, uniqueWord, totalWordPerCat = self.buildBagOfWord()
        countUniqueWord = len(uniqueWord)
        self.rule = {}

        for category in self.category:
            self.rule[category] = {}

        attrWeight = self.learnAttrWeight(attribute, category)
        for word in uniqueWord:
            for category in self.category:
                total = bagOfWord[category].get(word, 0)
                self.rule[category][word] = (total + 1) / (totalWordPerCat[category] + countUniqueWord)

    def buildBagOfWord(self):
        bagOfWord = {}
        uniqueWord = []
        totalWordPerCat = {}

        for index in range(0, len(self.attribute)):
            if bagOfWord.get(self.category[index]) is None:
                bagOfWord[self.category[index]] = {}
                totalWordPerCat[self.category[index]] = 0

            words = self.attribute[index].split()

            for word in words:
                if bagOfWord[self.category[index]].get(word) is None:
                    bagOfWord[self.category[index]][word] = 0

                bagOfWord[self.category[index]][word] += 1
                totalWordPerCat[self.category[index]] += 1

                if word not in uniqueWord:
                    uniqueWord.append(word)

        return bagOfWord, uniqueWord, totalWordPerCat

    def learnAttrWeight(self, attribute, category):
        pass
