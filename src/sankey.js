      google.charts.load('current', {'packages':['sankey']});
      google.charts.setOnLoadCallback(drawChart);

      function drawChart() {
        var data = new google.visualization.DataTable();
        data.addColumn('string', 'From');
        data.addColumn('string', 'To');
        data.addColumn('number', 'Weight');
        data.addRows([
          [ 'All node-pairs', 'Node labels', 1 ],
          [ 'All node-pairs', 'Node attributes', 1 ],
          
          [ 'Node histogram', 'Node labels', 1 ],
          [ 'Node histogram', 'Node attributes', 1 ],

					[ 'All edge-pairs', 'Edge labels', 1 ],
          [ 'All edge-pairs', 'Edge attributes', 1 ],
          
          [ 'Edge histogram', 'Edge labels', 1 ],
          [ 'Edge histogram', 'Edge attributes', 1 ],
          
          [ 'Shortest-path', 'Node labels', 1 ],
          [ 'Shortest-path', 'Node attributes', 1 ],
          
          [ 'GraphHopper', 'Node labels', 1 ],
          [ 'GraphHopper', 'Node attributes', 1 ],
		
    			[ 'Subtree pattern', 'Node labels', 1 ],
          
          [ 'Cyclic pattern', 'Node labels', 1 ],
          [ 'Cyclic pattern', 'Node attributes', 1 ],

					[ 'Graph edit distance', 'Node labels', 1 ],
          [ 'Graph edit distance', 'Node attributes', 1 ],
          [ 'Graph edit distance', 'Edge labels', 1 ],
          [ 'Graph edit distance', 'Edge attributes', 1 ],
					
          [ 'Graphlet', 'Edge attributes', 1 ],
          
          [ 'Direct product', 'Node labels', 1 ],
          [ 'Direct product', 'Node attributes', 1 ],

				 	[ 'Marginalized random walk', 'Node labels', 1 ],
          [ 'Marginalized random walk', 'Node attributes', 1 ],

					[ 'Random walk', 'Node labels', 1 ],
          [ 'Random walk', 'Node attributes', 1 ],
          [ 'Random walk', 'Edge attributes', 1 ],

					[ 'Quantum walk', 'Edge attributes', 1 ],
                    
          [ 'Weisfeiler-Lehman', 'Node labels', 1 ],
          [ 'Weisfeiler-Lehman', 'Edge labels', 1 ],

          [ 'Neighbourhood hash', 'Node labels', 1 ],
          [ 'Neighbourhood hash', 'Edge labels', 1 ],

          [ 'Neighbourhood subgraph pairwise distance', 'Node labels', 1 ],
          [ 'Neighbourhood subgraph pairwise distance', 'Edge labels', 1 ],

          [ 'Hadamard code', 'Node labels', 1 ],
          [ 'Hadamard code', 'Edge labels', 1 ],
          
          [ 'Propagation framework', 'Node labels', 1 ],
          [ 'Propagation framework', 'Node attributes', 1 ],
          [ 'Propagation framework', 'Edge labels', 1 ],
          [ 'Propagation framework', 'Edge attributes', 1 ],

          [ 'Message passing', 'Node labels', 1 ],
          [ 'Message passing', 'Node attributes', 1 ],

          [ 'Multiscale Laplacian', 'Node labels', 1 ],
          [ 'Multiscale Laplacian', 'Edge attributes', 1 ],

          [ 'Subgraph matching', 'Node labels', 1 ],
          [ 'Subgraph matching', 'Node attributes', 1 ],
          [ 'Subgraph matching', 'Edge labels', 1 ],
          [ 'Subgraph matching', 'Edge attributes', 1 ],

          [ 'Graph invariant framework', 'Node labels', 1 ],
          [ 'Graph invariant framework', 'Node attributes', 1 ],
          [ 'Graph invariant framework', 'Edge labels', 1 ],
          [ 'Graph invariant framework', 'Edge attributes', 1 ],

          [ 'Hash graph kernels', 'Node labels', 1 ],
          [ 'Hash graph kernels', 'Node attributes', 1 ],
          [ 'Hash graph kernels', 'Edge labels', 1 ],
          [ 'Hash graph kernels', 'Edge attributes', 1 ],

          [ 'Weighted decomposition', 'Node labels', 1 ],
          [ 'Weighted decomposition', 'Node attributes', 1 ],
          [ 'Weighted decomposition', 'Edge labels', 1 ],
          [ 'Weighted decomposition', 'Edge attributes', 1 ],

          [ 'Optimal assignment', 'Node labels', 1 ],
          [ 'Optimal assignment', 'Edge labels', 1 ],

          [ 'Deep graph kernels', 'Node labels', 1 ],
          [ 'Deep graph kernels', 'Edge labels', 1 ],

					[ 'Core based kernel framework', 'Node labels', 1 ],
          [ 'Core based kernel framework', 'Node attributes', 1 ],
          [ 'Core based kernel framework', 'Edge labels', 1 ],
          [ 'Core based kernel framework', 'Edge attributes', 1 ],


        ]);

        // Sets chart options.
        var options = {
          height: 2200,
          width: 1200,
          sankey: {
  link: {
    //color: {
     // fill: [
      //'#a6cee3',        // Custom color palette for sankey links.
      //'#1f78b4',        // Nodes will cycle through this palette
      //'#b2df8a',        // giving the links for that node the color.
     // '#33a02c'
    //],     // Color of the link.
      //fillOpacity: 0.8, // Transparency of the link.
      // stroke: 'black',  // Color of the link border.
      //strokeWidth: 1    // Thickness of the link border (default 0).
    //},
    colors: [
      '#1B9E77',        // Custom color palette for sankey links.
      '#7570B3',        // Nodes will cycle through this palette
      '#D95F02',        // giving the links for that node the color.
      '#E6AB02'
    ],
    colorMode: 'target',
  },
  node: {
  	fill: ['#1B9E77',        // Custom color palette for sankey links.
      '#7570B3',        // Nodes will cycle through this palette
      '#D95F02',        // giving the links for that node the color.
      '#E6AB02'],
      
  	colors: ['#1B9E77',        // Custom color palette for sankey links.
      '#7570B3',        // Nodes will cycle through this palette
      '#D95F02',        // giving the links for that node the color.
      '#E6AB02'],
 		nodePadding: 35,
    label: {
      fontName: 'Avenir',
      bold: true, 
      fontSize: 12,
  
      
  },
  	labelPadding: -190,
    width: 200,

 }
}
 
        };

        // Instantiates and draws our chart, passing in some options.
        var chart = new google.visualization.Sankey(document.getElementById('sankey_basic'));
        chart.draw(data, options);
      }



<html>
   <head>
      <title>Google Charts Tutorial</title>
      <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
   </head>
   
   <body>
       <div id="sankey_basic" style="width: 900px; height: 300px;"></div>
      <script language = "JavaScript">
               google.charts.load('current', {'packages':['sankey']});
      google.charts.setOnLoadCallback(drawChart);

      function drawChart() {
        var data = new google.visualization.DataTable();
        data.addColumn('string', 'From');
        data.addColumn('string', 'To');
        data.addColumn('number', 'Weight');
        data.addRows([
          [ 'All node-pairs', 'Node labels', 1 ],
          [ 'All node-pairs', 'Node attributes', 1 ],
          
          [ 'Node histogram', 'Node labels', 1 ],
          [ 'Node histogram', 'Node attributes', 1 ],

					[ 'All edge-pairs', 'Edge labels', 1 ],
          [ 'All edge-pairs', 'Edge attributes', 1 ],
          
          [ 'Edge histogram', 'Edge labels', 1 ],
          [ 'Edge histogram', 'Edge attributes', 1 ],
          
          [ 'Shortest-path', 'Node labels', 1 ],
          [ 'Shortest-path', 'Node attributes', 1 ],
          
          [ 'GraphHopper', 'Node labels', 1 ],
          [ 'GraphHopper', 'Node attributes', 1 ],
		
    			[ 'Subtree pattern', 'Node labels', 1 ],
          
          [ 'Cyclic pattern', 'Node labels', 1 ],
          [ 'Cyclic pattern', 'Node attributes', 1 ],

					[ 'Graph edit distance', 'Node labels', 1 ],
          [ 'Graph edit distance', 'Node attributes', 1 ],
          [ 'Graph edit distance', 'Edge labels', 1 ],
          [ 'Graph edit distance', 'Edge attributes', 1 ],
					
          [ 'Graphlet', 'Edge attributes', 1 ],
          
          [ 'Direct product', 'Node labels', 1 ],
          [ 'Direct product', 'Node attributes', 1 ],

				 	[ 'Marginalized random walk', 'Node labels', 1 ],
          [ 'Marginalized random walk', 'Node attributes', 1 ],

					[ 'Random walk', 'Node labels', 1 ],
          [ 'Random walk', 'Node attributes', 1 ],
          [ 'Random walk', 'Edge attributes', 1 ],

					[ 'Quantum walk', 'Edge attributes', 1 ],
                    
          [ 'Weisfeiler-Lehman', 'Node labels', 1 ],
          [ 'Weisfeiler-Lehman', 'Edge labels', 1 ],

          [ 'Neighbourhood hash', 'Node labels', 1 ],
          [ 'Neighbourhood hash', 'Edge labels', 1 ],

          [ 'Neighbourhood subgraph pairwise distance', 'Node labels', 1 ],
          [ 'Neighbourhood subgraph pairwise distance', 'Edge labels', 1 ],

          [ 'Hadamard code', 'Node labels', 1 ],
          [ 'Hadamard code', 'Edge labels', 1 ],
          
          [ 'Propagation framework', 'Node labels', 1 ],
          [ 'Propagation framework', 'Node attributes', 1 ],
          [ 'Propagation framework', 'Edge labels', 1 ],
          [ 'Propagation framework', 'Edge attributes', 1 ],

          [ 'Message passing', 'Node labels', 1 ],
          [ 'Message passing', 'Node attributes', 1 ],

          [ 'Multiscale Laplacian', 'Node labels', 1 ],
          [ 'Multiscale Laplacian', 'Edge attributes', 1 ],

          [ 'Subgraph matching', 'Node labels', 1 ],
          [ 'Subgraph matching', 'Node attributes', 1 ],
          [ 'Subgraph matching', 'Edge labels', 1 ],
          [ 'Subgraph matching', 'Edge attributes', 1 ],

          [ 'Graph invariant framework', 'Node labels', 1 ],
          [ 'Graph invariant framework', 'Node attributes', 1 ],
          [ 'Graph invariant framework', 'Edge labels', 1 ],
          [ 'Graph invariant framework', 'Edge attributes', 1 ],

          [ 'Hash graph kernels', 'Node labels', 1 ],
          [ 'Hash graph kernels', 'Node attributes', 1 ],
          [ 'Hash graph kernels', 'Edge labels', 1 ],
          [ 'Hash graph kernels', 'Edge attributes', 1 ],

          [ 'Weighted decomposition', 'Node labels', 1 ],
          [ 'Weighted decomposition', 'Node attributes', 1 ],
          [ 'Weighted decomposition', 'Edge labels', 1 ],
          [ 'Weighted decomposition', 'Edge attributes', 1 ],

          [ 'Optimal assignment', 'Node labels', 1 ],
          [ 'Optimal assignment', 'Edge labels', 1 ],

          [ 'Deep graph kernels', 'Node labels', 1 ],
          [ 'Deep graph kernels', 'Edge labels', 1 ],

					[ 'Core based kernel framework', 'Node labels', 1 ],
          [ 'Core based kernel framework', 'Node attributes', 1 ],
          [ 'Core based kernel framework', 'Edge labels', 1 ],
          [ 'Core based kernel framework', 'Edge attributes', 1 ],


        ]);

        // Sets chart options.
        var options = {
          height: 2000,
          width: 1000,
          sankey: {
  link: {
    //color: {
     // fill: [
      //'#a6cee3',        // Custom color palette for sankey links.
      //'#1f78b4',        // Nodes will cycle through this palette
      //'#b2df8a',        // giving the links for that node the color.
     // '#33a02c'
    //],     // Color of the link.
      //fillOpacity: 0.8, // Transparency of the link.
      // stroke: 'black',  // Color of the link border.
      //strokeWidth: 1    // Thickness of the link border (default 0).
    //},
    colors: [
      '#1B9E77',        // Custom color palette for sankey links.
      '#7570B3',        // Nodes will cycle through this palette
      '#D95F02',        // giving the links for that node the color.
      '#E6AB02'
    ],
    colorMode: 'target',
  },
  node: {
  	fill: ['#1B9E77',        // Custom color palette for sankey links.
      '#7570B3',        // Nodes will cycle through this palette
      '#D95F02',        // giving the links for that node the color.
      '#E6AB02'],
      
  	colors: ['#1B9E77',        // Custom color palette for sankey links.
      '#7570B3',        // Nodes will cycle through this palette
      '#D95F02',        // giving the links for that node the color.
      '#E6AB02'],
 		nodePadding: 25,
    label: {
      fontName: 'Avenir',
      bold: true, 
      fontSize: 12,
  
      
  },
  	// labelPadding: -190,
    // width: 200,

 }
}
 
        };

        // Instantiates and draws our chart, passing in some options.
        var chart = new google.visualization.Sankey(document.getElementById('sankey_basic'));
        chart.draw(data, options);
      }
      </script>
   </body>
</html>
