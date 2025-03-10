import { SvelteComponent } from "svelte";
import type { Gradio, SelectData } from "@gradio/utils";
declare const __propDef: {
    props: {
        row: boolean;
        is_container: boolean;
        component_type: string;
        var_name: string;
        active?: boolean | undefined;
        function_mode?: boolean | undefined;
        event_list: string[];
        is_input?: boolean | undefined;
        is_output?: boolean | undefined;
        triggers?: string[] | undefined;
        gradio: Gradio<{
            select: SelectData;
        }>;
    };
    events: {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type IndexProps = typeof __propDef.props;
export type IndexEvents = typeof __propDef.events;
export type IndexSlots = typeof __propDef.slots;
export default class Index extends SvelteComponent<IndexProps, IndexEvents, IndexSlots> {
}
export {};
