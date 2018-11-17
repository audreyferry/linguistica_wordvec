function oct2018_colorByCluster() {
  var EigSpreadsheetUrl = "https://docs.google.com/spreadsheets/d/1W_Bt-cy2XWp5v5rpeWTN70BeCBQXZYfYz763SHRJXzc/edit#gid=1737755252"
  numEigs = 3
  // var initialCol = 24;  // n=6:24;  n=12:36;  n=21:54
  // NOTE  row and column indexing starts at 1

  var LUT = [
    '#646464',    //nice charcoal
    '#96addb',    //blue violet
    '#c0c0c0',    //silver
    '#ff00ff',    //fuchsia
    '#2c8840',    //deeper green
    '#5ed1b7',    //nice aqua
    '#0000ff',  //blue
    '#ffff00',  //yellow
    '#00ffff',  //aqua
    '#800000',  //maroon
    '#008000',  //green
    '#6666ff',  //was navy  #000080   
    '#808000',  //olive
    '#800080',  //purple
    '#008080',  //teal
    '#808080',  //gray
    '#00ff00',  //lime
    '#8B4513',  //saddlebrown
    '#d15eb7',
    '#2c5088',
    '#004040',
    '#400040',
    //'#ff0000',  //red
    //'#fa3fc9',  //pinker fuchsia

  ];
  

  var ss = SpreadsheetApp.openByUrl(EigSpreadsheetUrl);
  var sheets = ss.getSheets();
  
  // sheet0
  //var range = sheets[0].getRange(2, 3, 6856)
  var range = sheets[0].getRange(2, 3, 7000)
  var lastRow = range.getLastRow()
  var values = range.getValues()
  var eigColoring = new Array(range.length)
  for (var i in values) {
    eigColoring[i] = [LUT[values[i][0]]]
  }
  range.setBackgrounds(eigColoring)
    
  for (j=1; j<=2; j++)  //sheets
    for (k = 1; k<2*numEigs; k++) {
      //                          (row, col, #rows)  
      var range = sheets[j].getRange(2, 5*k, lastRow)   //2d:  6856 x 1   It's an array of arrays 
      var values = range.getValues()                 //2d   matches range
      var eigColoring = new Array(range.length)      //1d
      for (var i in values) {
        eigColoring[i] = [LUT[values[i][0]]]         //2d   must match range
      }
    
    range.setBackgrounds(eigColoring);
    }
  return   
}

