"""
A data dashboard app of pyrdadionomics data file with a clinical file

Author: Larissa Voshol
Date: 22/01/2024
study: Bioinformatics
version: 1
"""
#imports
import numpy as np
import pandas as pd
import panel as pn
import bokeh as bk
import holoviews as hv
import hvplot.pandas
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, linear_cmap
from bokeh.models import HoverTool, ColumnDataSource, Whisker, Label
from io import StringIO

pn.extension()



#main dataset
temp_df = pd.DataFrame()

#options for group choises
df_radionmics = []
options = []
options_group = []

#Functions

def create_scater_plot(z, title,  df_clini, df_radionmics):
    """
    make a figure object ready for a scatter plot, with the according colormap and dataframe.

    args:
        z: the group the user has chosen
        title: a possible title the user wants to add
        df_clini: the dataframe with the clinical data
        df_radionomics: the dataframe with the pyradionomics
    
    returns:
        p: figure object to be translated to a scatter plot
        color: color map that has the length of z and fits the df
        df: dataframe thats used for the scatter plot

    """
    if z == "options":
        return
    try:
        #grab group column
        group = df_clini.loc[:,z]
        #unique factors
        uniq = group.unique()

        if z not in df_radionmics: 
            #paste to the frame with the data for the plot
            group = [str(item) for item in group]
            df_radionmics.insert(0, z, group)
        #order data ascending
        df = df_radionmics.sort_values(by=[z], ascending=True)

        # Create figure object
        TOOLS = "reset,tap,save"
        p = figure(title = title, tools=TOOLS)

        #create colormap for different lengths
        if len(uniq) == 1:
            color = "#1f77b4"
        elif len(uniq) <= 2:
            uniq = [str(item) for item in uniq]
            color = factor_cmap(z, ["#1f77b4", "#aec7e8"], uniq)
        elif len(uniq) <= 20:
            uniq = [str(item) for item in uniq]
            color = factor_cmap(z, bk.palettes.Category20[len(uniq)], uniq)
        elif len(uniq) > 20:
            # create how big the bins will be if you take 20 colors
            max_value = int(uniq.max())
            min_value = int(uniq.min())
            step = max(1, 3*int(round((max_value - min_value)/10, 0)))
            bins = list(range(min_value, max_value + step, step))
            # create the groups and append to dataframe
            new_z = []
            count=0
            for x in group:
                count +=1
                count_value = 0  
                for value in bins[:-1]:
                    if bins[count_value] >= float(x) and float(x) < bins[count_value + 1]:
                        new_z.append(bins[count_value])
                        break  
                    count_value += 1
            
            # add to datafram by removing the old first
            df.drop([z], axis=1 ,inplace=True)
            df.insert(0, z, new_z)
            df = df.sort_values(by=[z], ascending=True)
            #make colormap
            color = linear_cmap(z, bk.palettes.Category20[10], low=min(bins), high=max(bins))
    except:
        # create an empty plot
        p = figure(width=300, height=300, x_range=(0,1), y_range=(0,1))
        # add text in the middle of the plot
        text = Label(x=0.5, y=0.5, text='Upload your correct file', text_align='center', text_baseline='middle')
        p.add_layout(text)
        # make the other variables
        color = []
        df = pd.DataFrame()
        return p, color, df
    return p, color, df

def create_plot_heat( x, all, title, df_radionmics):
    """
    make a heatmap made from the wanted features

    args:
        x: the features you want to see.
        all: is the value of the select all features in the body
        title: a possible title the user wants to add
        df_radionomics: the dataframe with the pyradionomics

    returns: 
        p: the heatmap object that will be shown in the body
    """
    #grab the correct collums according to the widget
    temp_df = {}
    for element in x:
        temp = df_radionmics[element]
        temp_df[element] = temp
    df = pd.DataFrame(data = temp_df)
    
    # if the all checkbox is selected
    if all: 
        df = df_radionmics
    
    # Create the correlation df
    df_hm = df.corr()

    # Make the plot
    p = df_hm.hvplot.heatmap(title= title, 
                    colorbar=False, 
                    cmap="fire_r", 
                    xlabel="Feature 1",
                    ylabel= "Feature 2",
                    height = 700,
                    width = 700)
    p.opts(xrotation = 45,
        tools=["hover"])

    return p

def create_plot_f1(x, z, title,  df_clini, df_radionmics):
    """
    Makes a scatter plot with 1 adjustable feature

    args:
        x: The chosen feature the usere wants to see
        z: the group the user has chosen
        title: a possible title the user wants to add
        df_clini: the dataframe with the clinical data
        df_radionomics: the dataframe with the pyradionomics

    returns:
        p: the figure object that will be displayed in the body
    
    """
    
    #create figure and prep df and colorpallete 
    p, color, df = create_scater_plot(z, title,  df_clini, df_radionmics)

    # if df not right
    if not color:
        return p

    p.scatter("patient", x, source=df, color=color, legend_field=z)
    hover = HoverTool(tooltips=[("Patient ID", "@patient"),
    ("y-waarde","$x")])
    p.add_tools(hover)
    p.xaxis.axis_label = "patient"
    p.yaxis.axis_label = x
    return p

def create_plot_f2(x, y, z, title, df_clini, df_radionmics):
    """
    Makes a scatter plot with 2 adjustable feature

    args:
        x: The chosen feature the user wants to see
        y: the chosen feature the user also wants to see
        z: the group the user has chosen
        title: a possible title the user wants to add
        df_clini: the dataframe with the clinical data
        df_radionomics: the dataframe with the pyradionomics

    returns:
        p: the figure object that will be displayed in the body
    
    """
    #create figure and prep df and colorpallete 
    p, color, df = create_scater_plot(z, title,  df_clini, df_radionmics)
    # if df not right
    if not color:
        return p
    
    #create the plot
    p.scatter(x, y, source=df, color=color, legend_field=z)
    hover = HoverTool(tooltips=[("Patient ID", "@patient"),
    ("y-waarde","$x"),
    ("x-waarde","$x")])
    #adding of the hover tool
    p.add_tools(hover)
    p.xaxis.axis_label = x
    p.yaxis.axis_label = y

    return p

def create_boxplot(df_shape, features):
    """
    Makes a boxplot of eacht of the given values

    args:
        features: The chosen features the usere wants to see
        df_shape: the dataframe with the pyradionomics

    returns:
        p: the figure object that will be displayed in the body
    
    """
    #grab the correct collums according to the widget
    temp_df = {}
    for element in features:
        temp = df_shape[element]
        temp_df[element] = temp
    df_shape = pd.DataFrame(data = temp_df)

    # apply normalization techniques 
    for column in df_shape.columns: 
        df_shape[column] = df_shape[column] / df_shape[column].abs().max() 

    #unique keys
    kinds = list(df_shape.columns)

    #create quantile
    qs = df_shape.quantile([0.25, 0.5, 0.75])
    qs.insert(0,"new_index", ["q1", "q2", "q3"])
    temp_qs = qs.set_index("new_index")


    #create lower and upper
    upper=["upper"]
    lower=["lower"]
    for column in temp_qs:
        q1 = temp_qs.loc["q1",column]
        q3 = temp_qs.loc["q3",column]
        iqr = q3 - q1
        upper.append(q3 + 1.5*iqr)
        lower.append(q1 - 1.5*iqr)
    df_shape.reset_index(inplace=True)
    melt = pd.melt(df_shape, id_vars=['patient'], value_vars=df_shape.columns)


    #add to df
    qs.loc[len(qs.index)] = upper
    qs.loc[len(qs.index)] = lower
    qs.set_index("new_index", inplace=True)

    #switch columns with index
    New_df=qs.T.groupby(level=0).agg(lambda x : x.values.tolist()).stack().apply(pd.Series).unstack().sort_index(level=1,axis=1)
    New_df.columns=New_df.columns.droplevel(level=0)
    New_df.reset_index(inplace=True)

    #merge the quantile etc with the data
    df_boxplot = pd.merge(melt, New_df, left_on="variable", right_on="index", how="right").drop("index", axis=1)

    source = ColumnDataSource(df_boxplot)

    p = figure(x_range=kinds, tools="hover",
            title="Distribution with each feature",
            background_fill_color="#eaefef", y_axis_label="The unit of the feature", x_axis_label ="Features")

    # outlier range
    whisker = Whisker(base="variable", upper="upper", lower="lower", source=source)
    whisker.upper_head.size = whisker.lower_head.size = 20
    p.add_layout(whisker)

    # quantile boxes
    cmap = factor_cmap("variable", "TolRainbow7", kinds)
    p.vbar("variable", 0.7, "q2", "q3", source=source, color=cmap, line_color="black")
    p.vbar("variable", 0.7, "q1", "q2", source=source, color=cmap, line_color="black")

    # outliers
    outliers = df_boxplot[~df_boxplot.value.between(df_boxplot.lower, df_boxplot.upper)]
    p.scatter("variable", "value", source=outliers, size=6, color="black", alpha=0.3)
    p.axis.major_label_text_font_size="14px"
    p.axis.axis_label_text_font_size="12px"
    p.axis.major_label_orientation = 45

    return p


def add_graph_to_layout(items, event):
    """
    change the visibility of the graph select box in the body

    args:
        items: a list with the items that were given with the .link actions
            contains:
                add_graphbox: an body object that contains the select graph widget
                selct_graphtype: an widget where you can select the graph type you want to add
                graph_layout: list with the order of the graphs in the body  
        event: the add graph button widget, btn_add_graph
    """
    add_graphbox = items[0]
    select_graphtype = items[1]
    graph_layout = items[2]
    if not event:
        return
    # make invisible
    if add_graphbox.visible:
        add_graphbox.visible = False
    else:
        #make visibele
        select_graphtype.value = "Options"
        select_graphtype.disabled_options = reset_disabledoptions(graph_layout)
        add_graphbox.visible = True

def reset_disabledoptions(graph_layout):
    """
    reset the diseabled options in the select graph type widget. After a possible deletion of a graph.
    recognise the end of the name of the delete button, so it knows which one is deleted.

    args:
        graph_layout: list with the order of the graphs in the body  
    returns: list with the correct layout of the graphs
    """
    disable = []
    # look at all the graphs in the body at the moment
    for graph in graph_layout:
        if graph == "scatter with 1 feature":
            disable.append("Scatter(1f)")
        elif graph == "scatter with 2 features":
            disable.append("Scatter(2f)")
        elif graph == "Heatmap":
            disable.append("Heatmap")
        elif graph == "Boxplot":
            disable.append("Boxplot")
    # added the exisiting plots to the disable function
    return disable
            


def graphtype_disabledoptions(select_graphtype):
    """
    add a graph to the disabledoptions

    args:
        select_graphtype: select widget of the graph types
    """
    #grab the already disabled graph types
    disablegraphs = []
    alreadydisabled = select_graphtype.disabled_options
    for element in alreadydisabled:
        disablegraphs.append(element)

    #add the new one
    value = select_graphtype.value
    select_graphtype.value = "Options"
    if value != "Options":
        disablegraphs.append(value)
    
    # set value
    select_graphtype.disabled_options = disablegraphs
    return

def show_graphtype_in_body(items , event):
    """
    add the selected graphtype with the widgets to the body

    args:
        itmes: list with the given items
            contains:   
                add_graphbox:a body element which contains all elements that you can add a graph
                select_graphtype: the select widget for selecting the graphs
                dashboard_body: a list wath contains the body of the dashboard
                graph_layout: a list with the order of the graphs
        event: the select grapht type widget, select_graphtype 
    """
    #unpack variables
    add_graphbox = items[0]
    select_graphtype = items[1]
    dashboard_body = items[2]
    graph_layout = items[3]

    if not event:
        return
    
    #add the selected graph type to body
    graphtype = event.new
    if graphtype == "Options":
        return 
    elif graphtype == "Scatter(1f)":
        #to keep the overview of the plots
        graph_layout.append("scatter with 1 feature")
        # the new body
        scatter1f = pn.Row(pn.layout.Divider(),
        pn.Column("### Create your scatter plot with 1 value", 
                pn.WidgetBox(input_title, autoinput_f1, select_group, height = 500),
                btn_delete_f1
                ),
        pn.Column(fig_f1)
        )
        # append to body
        dashboard_body.append(scatter1f)
        dashboard_body.object = dashboard_body 
    elif graphtype == "Heatmap":
        #to keep the overview of the plots
        graph_layout.append("Heatmap")
        # the new body
        heatmap = pn.Row(pn.layout.Divider(),
        pn.Column("### Create your heatmap of the correlation between the features", 
                pn.WidgetBox(input_title_hm, multi_choice, checkbox_hm, height = 500),
                btn_delete_hm   
                ),
        pn.Column(fig_hm)
        )
        # append to body
        dashboard_body.append(heatmap)
        dashboard_body.object = dashboard_body 
    elif graphtype == "Scatter(2f)":
        #to keep the overview of the plots
        graph_layout.append("scatter with 2 features")
        # the body
        scatter2f = pn.Row(pn.layout.Divider(),
            pn.Column( "### Create your scatter plot with 2 values",
                      pn.WidgetBox(input_title_f2, autoinput_f2_x, autoinput_f2_y, select_group_f2, height = 500), 
                      btn_delete_f2
                      ),
            pn.Column(fig_f2)
        )
        #append to the body
        dashboard_body.append(scatter2f)
        dashboard_body.object = dashboard_body 
    elif graphtype == "Boxplot":
        #to keep the overview of the plots
        graph_layout.append("Boxplot")
        # the new body
        boxplot = pn.Row(pn.layout.Divider(),
            pn.Column("### Create your boxplots of the distribution from the features", 
                    pn.WidgetBox(multi_choice_bp, height = 500),
                    btn_delete_bp 
                    ),
            pn.Column(fig_bp)
            )
        # append to body
        dashboard_body.append(boxplot)
        dashboard_body.object = dashboard_body 
        return

    #make invisible
    add_graphbox.visible = False
    graphtype_disabledoptions(select_graphtype)
        
    return

def read_inputfiles(file, radionmics = False):
    """
    decode the string you get in through the input widget.
    args:
        file: the string containing the file
        radionmics: value if the file is a radiomics file
    
    return:
        df: dataframe containg the file
    """
    # transform the file to a df
    df = pd.read_csv(StringIO(file.decode("utf-8")))
    if radionmics:
        df.pop("Unnamed: 0")
    df.set_index("patient", inplace=True)

    return df


def file_radionomics(df_widget, event):
    """
    grab the file input of the input widget and create a dataframe

    args:   
        df_widget: input widget
        event: the radionomics file input widget, radionomics_input
    """
    # current file
    file = event.new
    # the new file is different than the old
    if file != event.old:
        df = read_inputfiles(file, radionmics = True)

        # edit the widget
        df_widget.value = df
        # df_widget.visible = True
        return
    return 

def file_clinical(df_widget, event):
    """
    grab the file input of the input widget and create a dataframe

    args:   
        df_widget: input widget
        event: the radionomics file input widget, clinical_input
    """
    # current file
    file = event.new
    # the new file is different than the old
    if file != event.old:
        df = read_inputfiles(file)

        # edit the widget
        df_widget.value = df
        # df_widget.visible = True
        return 
    return 

def delete_graph(items, event):
    """
    delete the graph from the body 

    args:
        items: list with the items given in the .link object
            contains:
            dashboard_body: list with the body
            graph_layout: list with the order of the graphs in the body
    """
    #unpack items
    dashboard_body = items[0]
    graph_layout = items[1]

    #make a index of the order of the graphlayout
    locations = dict(enumerate(graph_layout))
    #grab the graptype you want to delete
    button = event.obj.description[26:]
    for loc, graphtype in locations.items():
        #got the loc of the graphtype you want to delete
        if graphtype == button:
            dashboard_body.pop(loc)
            graph_layout.remove(graphtype)

def select_features(dataframe_radio, event):
    """
    select the group of features you want to use in the plots
    args:
        dataframe_radio: list with items given with the .link object
            contains:
                df: dataframe of radionomics
                btn: btn_add_graph widget
        event: multi choice widget for the features, multi_choice_df
    """
    #standard feature values
    feature_indices = {"shape features": 23, "first order features": 38, "GLCM features": 55,
                         "GLDM features": 79, "GLRLM features": 93, "GLZSM_features": 109, "NGTDM_features": 125}
    indices = [23 ,38 ,55, 79, 93, 109, 125]

    # the selected values
    features = event.new
    df = dataframe_radio[0].value
    df_old = pd.DataFrame()

    btn = dataframe_radio[2] 

    # make a dataframe of the features you wish to see 
    for feature in features:
        indx = feature_indices[feature]
        if indx == 125:
            if not df_old.empty:
                df_new = df.iloc[:,indx:]
                #combine the 2 different features
                df_old = pd.concat([df_old, df_new], axis=1 )
            else:
                df_old = df.iloc[:,indx:]
        else:
            if not df_old.empty:
                numb = [i for i,x in enumerate(indices) if x == indx]
                df_new = df.iloc[:, indx: indices[numb[0]+ 1] -1]
                #combine the 2 different features
                df_old = pd.concat([df_old, df_new], axis=1)
            else:
                #aangegeven feature tot the volgende 
                numb = [i for i,x in enumerate(indices) if x == indx]
                df_old = df.iloc[:, indx: indices[numb[0]+ 1] -1]
    
    #change the col and index names
    ticks_names = list(df_old.columns)
    ticksnames = {}
    for names in ticks_names:
        ticksnames[names] = names.replace("_", " ")
    df_old.rename(index = ticksnames, columns= ticksnames, inplace=True)

    #make new dataframe visible
    dataframe_radio[1].value = df_old
    dataframe_radio[1].visible = True

    #make button visible
    btn.visible = True
    
    return df_old

def change_options(items, event):
    """
    change the options for the widgets to the input file

    args:
        items: list with elementen given through the .link object
            contains:
                autoinput_f1 = autoinput widget for 1 feature in scatter 1f
                autoinput_f2_x = autoinput widget for 1 feature in scatter 2f
                autoinput_f2_y = autoinput widget for 1 feature in scatter 2f
                multi_choice = multichoice widget for features for heatmap
                multi_choice_bp = multichoice widget for features for boxplot
        event: input file widget of pyradiomics, pn_df_edit
    """
    # unpack items
    autoinput_f1 = items[0]
    autoinput_f2_x = items[1] 
    autoinput_f2_y = items[2]
    multi_choice = items[3]
    multi_choice_bp = items[4]
    
    #make the options based of the dataframe in the widget
    dataframe = event.new
    options = list(dataframe.columns)

    # give the options
    autoinput_f1.options = options
    autoinput_f2_x.options = options
    autoinput_f2_y.options = options
    multi_choice.options = options

    # also set the first element as the value for the heatmap and boxplot
    multi_choice.value = [options[0]]
    multi_choice_bp.options = options
    multi_choice_bp.value = [options[0]]
    return

def change_options_group(items, event):
    """
    change the options for the widgets to the input file

    args:
        items: list with elementen given through the .link object
            contains:
                select_group: the select widget for selecting the element you want to group it to scatter f1
                select_group_f2: the select widget for selecting the element you want to group it to scatter f2
        event: input file widget of clinical, pn_df_clinical
    """
    # unpack items
    select_group = items[0]
    select_group_f2 = items[1]

    #make the options based of the dataframe in the widget
    dataframe = event.new
    # no nan containing columns
    df_nonan = dataframe.dropna(axis=1)
    # give the options
    select_group.options = list(df_nonan.columns)
    select_group_f2.options = list(df_nonan.columns)


# Dashboard
dashboard = pn.template.BootstrapTemplate(
    title= "PlotRadionomics",
    header_background = "indigo",
    sidebar=[]
)

#Create widgets
## main
btn_add_graph = pn.widgets.Button(
    name="Add graph", 
    icon="plus", 
    button_type="primary", 
    description="Push to add a another graph", 
    visible=False
    )
select_graphtype = pn.widgets.Select(
    name="Choose your graph type", 
    options=["Options", "Scatter(1f)", "Scatter(2f)", "Boxplot", "Heatmap"], 
    disabled_options=[], 
    value="Options"
    )
radionomics_input = pn.widgets.FileInput(accept=".csv")
clinical_input = pn.widgets.FileInput(accept=".csv")
multi_choice_df = pn.widgets.MultiChoice(name='Choose the features you want to compare',
    options=["shape features", "first order features", "GLCM features",
              "GLDM features", "GLRLM features", "GLZSM features", "NGTDM features"])
pn_df_clinical = pn.widgets.DataFrame(temp_df ,visible = False,  reorderable= True, width=500 )
pn_df_radionmics = pn.widgets.DataFrame(temp_df ,visible = False, reorderable= True, width=500)
pn_df_edit = pn.widgets.DataFrame(temp_df, visible= False, reorderable= True, width=900)


## Scatter(1f)
autoinput_f1 = pn.widgets.AutocompleteInput(
    name="Feature you want to see:", 
    options=options,
    case_sensitive=False, 
    search_strategy="includes",
    placeholder="Start to write feature"
    )
select_group = pn.widgets.Select(
    name="Based on wich group", 
    options=options_group,
    value = "Sex")
input_title = pn.widgets.TextInput(
    name="Graph title",
      placeholder="Enter a string here..."
      )
btn_delete_f1 = pn.widgets.Button(
    name="delete", 
    button_type="danger", 
    description="Push to delete the graph: scatter with 1 feature", 
)

## Heatmap
input_title_hm = pn.widgets.TextInput(
    name="Graph title",
      placeholder="Enter a string here..."
      )
multi_choice = pn.widgets.MultiChoice(name='Choose the features you want to compare',
    options=options,
    value = ["original shape Flatness"])
checkbox_hm = pn.widgets.Checkbox(name='All features')
btn_delete_hm = pn.widgets.Button(
    name="delete", 
    button_type="danger", 
    description="Push to delete the graph: Heatmap", 
)


## Scatter(2f)
autoinput_f2_x = pn.widgets.AutocompleteInput(
    name="Feature you want to see for x-axis", 
    options=options,
    case_sensitive=False, 
    search_strategy="includes",
    placeholder="Start to write feature"
    )
autoinput_f2_y = pn.widgets.AutocompleteInput(
    name="Feature you want to see for y-axis", 
    options=options,
    case_sensitive=False, 
    search_strategy="includes",
    placeholder="Start to write feature"
    )
select_group_f2 = pn.widgets.Select(
    name="Based on wich group", 
    options=options_group,
    value = "Sex"
    )
input_title_f2 = pn.widgets.TextInput(
    name="Graph title",
      placeholder="Enter a string here..."
      )
btn_delete_f2 = pn.widgets.Button(
    name="delete", 
    button_type="danger", 
    description="Push to delete the graph: scatter with 2 features", 
)

## boxplot
multi_choice_bp = pn.widgets.MultiChoice(name='Choose the features you want to see',
    options=options,
    value = ["original shape Flatness"])
btn_delete_bp = pn.widgets.Button(
    name="delete", 
    button_type="danger", 
    description="Push to delete the graph: Boxplot", 
)


#Bind widgets

## binding of the different figures with the widgets
fig_f1 = pn.bind(create_plot_f1, x=autoinput_f1, z=select_group, title=input_title, df_clini=pn_df_clinical, df_radionmics=pn_df_edit)
fig_f2 = pn.bind(create_plot_f2, x=autoinput_f2_x, y=autoinput_f2_y, z=select_group_f2, title=input_title,  df_clini=pn_df_clinical, df_radionmics=pn_df_edit)
fig_hm = pn.bind(create_plot_heat, title= input_title_hm, x=multi_choice, all=checkbox_hm,  df_radionmics=pn_df_edit)
fig_bp = pn.bind(create_boxplot,  df_shape=pn_df_edit, features=multi_choice_bp)

#Create body items
## contains the menu of what graph you want to select
add_graphbox = pn.WidgetBox(
    "## Create your graph", 
    select_graphtype, visible = False
    )

## welcome element
welcome = pn.Column(
    "# Welcome at PlotRadionomics",
    """Here you can take a look at your pyradiomics data using a boxplot, heatmap or a scatter plot.
    In the pyradiomics data you've different groups of data, you can take a look at the different
    groups by selecting the features underneath. At the moment you need to put in a pyradiomics 
    For this you also need a clinical file with data of all the same patients in your radionomics.
    This data can be info you can filter the dataset on.
    Only csv files are accepted.""",
    pn.Row("Upload your file with the radionomics data here", radionomics_input,
    "Upload your file with the clinical patient data here", clinical_input),
    pn.Row(pn_df_radionmics, pn_df_clinical),
    pn.Row(multi_choice_df),
    pn.Row(pn_df_edit))

#the body
body = pn.Column(pn.Column(
            welcome,
            "## Want to add a graph?", 
            "First upload the files and chose your features",
            btn_add_graph, add_graphbox
            ), css_classes=['flex-container'])

#list with all the graphtypes in the body
graph_layout = ["welcome"]

#watch the widgets
## add graph to the body
items = [add_graphbox, select_graphtype, body, graph_layout]
select_graphtype.link(items, callbacks={"value": show_graphtype_in_body})
## takes care of the graph select menu
items = [add_graphbox, select_graphtype, graph_layout]
btn_add_graph.link(items, callbacks={"value": add_graph_to_layout})
## takes care of the feature select menu
items = [pn_df_radionmics, pn_df_edit, btn_add_graph]
multi_choice_df.link(items, callbacks={"value": select_features})
## delete graphs of layout
items = [body, graph_layout]
btn_delete_f1.link(items, callbacks={"value": delete_graph})
btn_delete_f2.link(items, callbacks={"value": delete_graph})
btn_delete_hm.link(items, callbacks={"value": delete_graph})
btn_delete_bp.link(items, callbacks={"value": delete_graph})
## binding of the dataframes to the page
radionomics_input.link(pn_df_radionmics, callbacks={"value": file_radionomics})
clinical_input.link(pn_df_clinical, callbacks={"value": file_clinical})
## edit options in widgets
items = [select_group, select_group_f2]
pn_df_clinical.link(items, {"value": change_options_group})
items = [autoinput_f1, autoinput_f2_x, autoinput_f2_y, multi_choice, multi_choice_bp]
pn_df_edit.link(items, {"value": change_options})



dashboard.main.append(body)


dashboard.servable()

# error catching
# delete button werkende
# 1f grafiek
# zscore bij boxplot


stylesheet = """
flex-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100vh;  
}

"""