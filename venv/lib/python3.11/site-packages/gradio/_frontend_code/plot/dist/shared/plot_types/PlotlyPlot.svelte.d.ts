import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        value: any;
        show_label: boolean;
        loaded_plotly_css?: boolean | undefined;
    };
    events: {
        load: CustomEvent<undefined>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type PlotlyPlotProps = typeof __propDef.props;
export type PlotlyPlotEvents = typeof __propDef.events;
export type PlotlyPlotSlots = typeof __propDef.slots;
export default class PlotlyPlot extends SvelteComponent<PlotlyPlotProps, PlotlyPlotEvents, PlotlyPlotSlots> {
}
export {};
