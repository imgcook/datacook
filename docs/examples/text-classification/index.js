var app = new Vue({
  el: '#app',
  data: {
    curTab: 'inbox',
    messageList: ['a', 'b', 'c'],
    labels: [],
    count: 1,
    model: null,
    vectorizer: null
  },
  computed: {
    inboxList: function() {
      if (this.labels.length) {
        return this.messageList.filter((d, i) => this.labels[i] === 'ham');
      }
      return this.messageList;
    },
    trashList: function() {
      if (this.labels.length) {
        return this.messageList.filter((d, i) => this.labels[i] === 'spam');
      }
      return [];
    },
    curList: function() {
      if (this.curTab === 'trash') {
        return this.trashList;
      }
      return this.inboxList;
    }
  },
  mounted() {
    this.messageList = ['b', 'c']
    // let app = this;
    fetch('../../assets/dataset/spam.csv').then((res) => res.text())
      .then((res) => {
        const data = res.split('\n').map(d => d.split(',')[1]);
        this.messageList = data.slice(1,500);
    });
    fetch('./model.json').then((res) => res.text())
      .then((modelJson) => {
        const { MultinomialNB } = datacook.Model.NaiveBayes;
        const mnb = new MultinomialNB();
        mnb.load(modelJson);
        this.model = mnb;
    });
    fetch('./vectorizer.json').then((res) => res.text())
      .then((vectorizerJson) => {
        const { CountVectorizer } = datacook.Text;
        const vectorizer = new CountVectorizer();
        vectorizer.load(vectorizerJson);
        this.vectorizer = vectorizer;
    });
  },
  methods: {
    collectSpam: async function() {
      if (this.model && this.vectorizer) {
        console.log(this.model);
        console.log(this.vectorizer.wordOrder.length);
        const textVector = this.vectorizer.transform(this.messageList);
        const predY = await this.model.predict(textVector);
        console.log(this.vectorizer);
        const predYArray = predY.arraySync();
        console.log(JSON.stringify(predYArray));
        this.labels = predYArray;
      }
    },
    switchTab: function(tab) {
      this.curTab = tab;
    }
  }
});