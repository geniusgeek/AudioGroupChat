import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        show_fullscreen_button?: boolean | undefined;
        show_copy_button?: boolean | undefined;
        show_search?: ("none" | "search" | "filter") | undefined;
        is_fullscreen?: boolean | undefined;
        on_copy: () => Promise<void>;
        on_commit_filter: () => void;
        current_search_query?: (string | null) | undefined;
    };
    events: {
        click: MouseEvent;
        search: CustomEvent<string | null>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {};
};
export type ToolbarProps = typeof __propDef.props;
export type ToolbarEvents = typeof __propDef.events;
export type ToolbarSlots = typeof __propDef.slots;
export default class Toolbar extends SvelteComponent<ToolbarProps, ToolbarEvents, ToolbarSlots> {
}
export {};
