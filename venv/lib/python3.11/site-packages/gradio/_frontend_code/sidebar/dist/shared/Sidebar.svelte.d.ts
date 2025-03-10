import { SvelteComponent } from "svelte";
declare const __propDef: {
    props: {
        open?: boolean | undefined;
        width: number | string;
        position?: ("left" | "right") | undefined;
    };
    events: {
        expand: CustomEvent<void>;
        collapse: CustomEvent<void>;
    } & {
        [evt: string]: CustomEvent<any>;
    };
    slots: {
        default: {};
    };
};
export type SidebarProps = typeof __propDef.props;
export type SidebarEvents = typeof __propDef.events;
export type SidebarSlots = typeof __propDef.slots;
export default class Sidebar extends SvelteComponent<SidebarProps, SidebarEvents, SidebarSlots> {
}
export {};
